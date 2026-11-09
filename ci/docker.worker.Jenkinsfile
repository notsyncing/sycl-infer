// Jenkins pipeline: build the sycl-infer LLM worker image ONLY
// (docker/Dockerfile.worker).
//
// Use this when the worker image is deployed on its own or when the router
// runs elsewhere (the `docker/compose.yaml` topology).  For the combined
// "router spawns its workers" image, use ci/docker.Jenkinsfile with
// IMAGE_TARGET=all, or ci/docker.router.Jenkinsfile for the router alone.
//
// The agent only needs a Docker daemon.  The image carries the oneAPI
// toolchain and the GPU runtime; the agent needs no icpx and no GPU.
//
// AOT bakes the device image for one GPU SKU:
//   acm-g10  Arc A770 (Xe-HPG)
//   adl-p    Iris Xe (Xe-LP)
//   lnl      Arc 140V / Lunar Lake (Xe2)
// A JIT image (AOT=false) runs on any card but pays the runtime JIT.
//
// Tags: <python/VERSION>-<7-char hash>[.aot.<device>] - ci/Jenkinsfile's wheel
// version rule with '+' written '-' (a Docker tag cannot contain '+').  The
// AOT device is sanitized like the wheel's (acm-g10 -> acm.g10).  IMAGE_TAG
// overrides it; EXTRA_TAGS adds aliases such as latest.
pipeline {
    agent any

    environment {
        DOCKER_BUILDKIT = '1'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        // AOT lowers one device image per kernel at the final link.
        timeout(time: 12, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    parameters {
        booleanParam(name: 'AOT', defaultValue: true,
                     description: 'AOT-compile the device image instead of runtime JIT.')
        string(name: 'AOT_DEVICE', defaultValue: 'acm-g10',
               description: 'ocloc target: acm-g10 (Arc A770), adl-p (Iris Xe), lnl (Arc 140V / Lunar Lake).')
        string(name: 'ONEAPI_VERSION', defaultValue: '2026.2.0',
               description: 'oneAPI toolkit version. 2026.1.1 is ~16% faster for the 35B prefill (see AGENTS.md).')
        string(name: 'UBUNTU_TAG', defaultValue: 'ubuntu26.04',
               description: 'Ubuntu tag of the oneAPI base images (must match the host glibc).')
        string(name: 'DNNL_VERSION', defaultValue: '2026.1',
               description: 'oneDNN version bundled by the selected oneAPI toolkit.')
        string(name: 'RUNTIME_IMAGE', defaultValue: 'intel/oneapi-runtime',
               description: 'Runtime base image; use intel/oneapi-toolkit with INSTALL_GPU=false if the slim runtime is unavailable.')
        booleanParam(name: 'INSTALL_GPU', defaultValue: true,
                     description: 'Install the Level Zero driver / OpenCL ICD from the Intel graphics PPA.')
        string(name: 'UBUNTU_MIRROR', defaultValue: 'archive.ubuntu.com',
               description: 'Ubuntu mirror rewritten into the image sources at build time.')
        string(name: 'BUILD_JOBS', defaultValue: '',
               description: 'Parallel compile jobs inside the image; empty = nproc. Lower it on a low-memory agent.')

        string(name: 'IMAGE_NAMESPACE', defaultValue: 'sycl-infer',
               description: 'Image namespace/org: <registry>/<namespace>/worker.')
        string(name: 'REGISTRY', defaultValue: '',
               description: 'Registry host (and optional path), e.g. ghcr.io/acme. Empty = Docker Hub.')
        string(name: 'IMAGE_TAG', defaultValue: '',
               description: 'Tag override. Empty = derive it from python/VERSION + the 7-char git hash (+ .aot.<device> when AOT), the same rule as ci/Jenkinsfile. Docker forbids "+", so it is written "-" (0.1.0-<hash>[.aot.<device>]).')
        string(name: 'EXTRA_TAGS', defaultValue: '',
               description: 'Additional tags, comma-separated (e.g. latest).')
        booleanParam(name: 'PUSH_IMAGE', defaultValue: false,
                     description: 'Push the built image (requires REGISTRY).')
        string(name: 'REGISTRY_CREDENTIALS_ID', defaultValue: '',
               description: 'Jenkins username/password credential for docker login. Empty = assume already logged in.')

        booleanParam(name: 'RUN_SMOKE_TEST', defaultValue: true,
                     description: 'Run the image (--help) and assert the worker binary exists.')
        booleanParam(name: 'PULL_BASE_IMAGE', defaultValue: true,
                     description: 'Pass --pull so the base image is refreshed.')
        booleanParam(name: 'NO_CACHE', defaultValue: false,
                     description: 'Pass --no-cache (a clean rebuild, much slower).')
        string(name: 'PLATFORM', defaultValue: 'linux/amd64',
               description: 'Build platform; empty = the agent\'s platform. Intel GPU hosts are amd64.')
        string(name: 'HTTP_PROXY', defaultValue: '',
               description: 'Build-time proxy for apt/wget inside the image.')
        string(name: 'HTTPS_PROXY', defaultValue: '',
               description: 'Build-time HTTPS proxy for apt/wget inside the image.')
        string(name: 'NO_PROXY', defaultValue: 'localhost,127.0.0.1,::1',
               description: 'Proxy bypass list for the build.')
        string(name: 'EXTRA_BUILD_ARGS', defaultValue: '',
               description: 'Additional docker build flags, whitespace-separated.')
    }

    stages {
        stage('Prepare') {
            steps {
                checkout scm
                script {
                    def tagOverride = (params.IMAGE_TAG ?: '').trim()
                    if (tagOverride && !(tagOverride ==~ /[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}/)) {
                        error("IMAGE_TAG '${tagOverride}' is not a valid Docker tag")
                    }
                    def extraTags = (params.EXTRA_TAGS ?: '').trim()
                    if (extraTags) {
                        extraTags.split(/[,\s]+/).each { String t ->
                            if (!(t ==~ /[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}/)) {
                                error("EXTRA_TAGS entry '${t}' is not a valid Docker tag")
                            }
                        }
                    }
                    def ns = (params.IMAGE_NAMESPACE ?: '').trim().replaceAll('^/+', '').replaceAll('/+$', '')
                    if (!ns) {
                        error('IMAGE_NAMESPACE must not be empty')
                    }
                    def registry = (params.REGISTRY ?: '').trim().replaceAll('/+$', '')
                    if (params.PUSH_IMAGE && !registry) {
                        error('PUSH_IMAGE=true requires REGISTRY (e.g. ghcr.io/acme)')
                    }
                    def jobs = (params.BUILD_JOBS ?: '').trim()
                    if (jobs && !(jobs ==~ /[1-9][0-9]*/)) {
                        error("BUILD_JOBS must be a positive integer, got '${jobs}'")
                    }

                    env.GIT_SHA = sh(
                        returnStdout: true,
                        script: 'git rev-parse --short=7 HEAD 2>/dev/null || echo unknown'
                    ).trim()

                    // Same version rule as ci/Jenkinsfile:
                    //   <python/VERSION> + <7-char hash> [ + .aot.<device> ]
                    // A Docker tag cannot contain '+', so it is written '-'.
                    def baseVersion = readFile('python/VERSION').trim()
                    def aotDevice = ((params.AOT_DEVICE ?: '').trim() ?: 'unknown')
                        .replaceAll('[^A-Za-z0-9]+', '.')
                    env.VERSION_BASE = "${baseVersion}+${env.GIT_SHA}"
                    env.VERSION_AOT_SUFFIX = params.AOT ? ".aot.${aotDevice}" : ''
                    env.IMAGE_TAG_OVERRIDE = tagOverride
                    env.EXTRA_TAGS = extraTags
                    def version = "${env.VERSION_BASE}${env.VERSION_AOT_SUFFIX}"
                    env.IMAGE_TAG_VALUE = tagOverride ? tagOverride : version.replace('+', '-')
                    env.IMAGE_PREFIX = "${registry ? registry + '/' : ''}${ns}"

                    echo """
                        |sycl-infer worker image
                        |  image       : ${env.IMAGE_PREFIX}/worker:${env.IMAGE_TAG_VALUE}
                        |  version     : ${version}  (ci/Jenkinsfile rule)
                        |  extra tags  : ${extraTags ?: '-'}
                        |  git         : ${env.GIT_SHA}
                        |  AOT         : ${params.AOT} (device ${aotDevice})
                        |  oneAPI      : ${params.ONEAPI_VERSION} / ${params.UBUNTU_TAG}
                        |  oneDNN      : ${params.DNNL_VERSION}
                        |  runtime base: ${params.RUNTIME_IMAGE} (GPU stack: ${params.INSTALL_GPU})
                        |  build jobs  : ${jobs ?: 'nproc'}
                        |  push        : ${params.PUSH_IMAGE}
                        |  proxy       : ${(params.HTTP_PROXY ?: params.HTTPS_PROXY) ? 'set' : 'not set'}
                        |""".stripMargin()

                    sh 'docker version --format "{{.Server.Version}}" >/dev/null'
                }
            }
        }

        stage('Build worker image') {
            steps {
                script {
                    def tag = env.IMAGE_TAG_VALUE
                    def repo = "${env.IMAGE_PREFIX}/worker".toString()
                    def version = "${env.VERSION_BASE}${env.VERSION_AOT_SUFFIX}".toString()

                    def extraTags = (env.EXTRA_TAGS ?: '').split(/[,\s]+/).findAll { it }
                    def tags = ([tag] + extraTags).unique()

                    def created = sh(
                        returnStdout: true,
                        script: 'date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown'
                    ).trim()

                    def shQuote = { Object s ->
                        "'" + s.toString().replace("'", "'\\''") + "'"
                    }

                    def args = ['docker', 'build']
                    if (params.PULL_BASE_IMAGE) {
                        args << '--pull'
                    }
                    if (params.NO_CACHE) {
                        args << '--no-cache'
                    }
                    def platform = (params.PLATFORM ?: '').trim()
                    if (platform) {
                        args += ['--platform', platform]
                    }
                    args += ['-f', 'docker/Dockerfile.worker']

                    args += ['--build-arg', "ONEAPI_VERSION=${params.ONEAPI_VERSION}".toString()]
                    args += ['--build-arg', "UBUNTU_TAG=${params.UBUNTU_TAG}".toString()]
                    args += ['--build-arg', "DNNL_VERSION=${params.DNNL_VERSION}".toString()]
                    args += ['--build-arg', "RUNTIME_IMAGE=${params.RUNTIME_IMAGE}".toString()]
                    args += ['--build-arg', "AOT=${params.AOT ? 'ON' : 'OFF'}".toString()]
                    args += ['--build-arg', "AOT_DEVICE=${params.AOT_DEVICE}".toString()]
                    args += ['--build-arg', "INSTALL_GPU=${params.INSTALL_GPU ? '1' : '0'}".toString()]
                    args += ['--build-arg', "UBUNTU_MIRROR=${params.UBUNTU_MIRROR}".toString()]
                    def jobs = (params.BUILD_JOBS ?: '').trim()
                    if (jobs) {
                        args += ['--build-arg', "BUILD_JOBS=${jobs}".toString()]
                    }
                    def httpProxy = (params.HTTP_PROXY ?: '').trim()
                    def httpsProxy = (params.HTTPS_PROXY ?: '').trim()
                    def noProxy = (params.NO_PROXY ?: '').trim()
                    if (httpProxy) {
                        args += ['--build-arg', "HTTP_PROXY=${httpProxy}".toString()]
                    }
                    if (httpsProxy) {
                        args += ['--build-arg', "HTTPS_PROXY=${httpsProxy}".toString()]
                    }
                    if (noProxy) {
                        args += ['--build-arg', "NO_PROXY=${noProxy}".toString()]
                    }

                    tags.each { String t -> args += ['-t', "${repo}:${t}".toString()] }

                    args += ['--label', "org.opencontainers.image.revision=${env.GIT_SHA}".toString()]
                    args += ['--label', "org.opencontainers.image.version=${version}".toString()]
                    args += ['--label', "org.opencontainers.image.created=${created}".toString()]

                    def extra = (params.EXTRA_BUILD_ARGS ?: '').trim()
                    if (extra) {
                        extra.split(/\s+/).each { String a -> args << a }
                    }

                    args << '.'

                    sh(script: args.collect { shQuote(it) }.join(' '),
                       label: 'docker build worker')
                }
            }
        }

        stage('Smoke test') {
            when {
                expression { return params.RUN_SMOKE_TEST }
            }
            steps {
                script {
                    def shQuote = { Object s ->
                        "'" + s.toString().replace("'", "'\\''") + "'"
                    }
                    def img = "${env.IMAGE_PREFIX}/worker:${env.IMAGE_TAG_VALUE}".toString()

                    // --help exits before any SYCL device query.
                    sh "docker run --rm ${shQuote(img)} --help >/dev/null"
                    sh "docker run --rm --entrypoint sh ${shQuote(img)} -c " +
                       shQuote('test -x /usr/local/bin/sycl-infer')
                }
            }
        }

        stage('Push image') {
            when {
                expression { return params.PUSH_IMAGE }
            }
            steps {
                script {
                    def registry = (params.REGISTRY ?: '').trim().replaceAll('/+$', '')
                    if (!registry) {
                        error('PUSH_IMAGE=true requires REGISTRY')
                    }
                    def extraTags = (env.EXTRA_TAGS ?: '').split(/[,\s]+/).findAll { it }
                    def tags = ([env.IMAGE_TAG_VALUE] + extraTags).unique()

                    def shQuote = { Object s ->
                        "'" + s.toString().replace("'", "'\\''") + "'"
                    }
                    def host = registry.split('/')[0]
                    def credentialsId = (params.REGISTRY_CREDENTIALS_ID ?: '').trim()
                    if (credentialsId) {
                        withCredentials([usernamePassword(
                                credentialsId: credentialsId,
                                usernameVariable: 'REG_USER',
                                passwordVariable: 'REG_PASS')]) {
                            sh 'printf %s "$REG_PASS" | docker login ' +
                               shQuote(host) + ' --username "$REG_USER" --password-stdin'
                        }
                    } else {
                        echo "REGISTRY_CREDENTIALS_ID is empty; assuming the agent is already logged in to ${host}"
                    }

                    def repo = "${env.IMAGE_PREFIX}/worker".toString()
                    tags.each { String t ->
                        def pushed = "${repo}:${t}".toString()
                        sh "docker push ${shQuote(pushed)}"
                    }
                }
            }
        }
    }

    post {
        success {
            echo "Built worker image ${env.IMAGE_TAG_VALUE} from ${env.GIT_SHA} (version ${env.VERSION_BASE}${env.VERSION_AOT_SUFFIX})."
        }
    }
}
