// Build the deployment images for sycl-infer.
//
// The agent only needs a working Docker daemon: the images carry their own
// toolchain, and the AOT device lowering happens inside the build stage.  No
// GPU, icpx, oneDNN or oneAPI installation is required on the agent.
//
// Targets:
//   all            docker/Dockerfile.all     router + LLM worker + ONNX worker
//                                            (the "router spawns its workers"
//                                            topology; ENTRYPOINT is the router)
//   worker         docker/Dockerfile.worker  the SYCL LLM worker
//   router         docker/Dockerfile.router  the plain host router (no SYCL,
//                                            no oneAPI, no GPU)
//   worker+router  build the two separate images (not the combined one)
//
// AOT: -DSYCL_INFER_AOT bakes the device image for ONE GPU SKU.  Match
// AOT_DEVICE to the deployment card:
//   acm-g10  Arc A770 (Xe-HPG)
//   adl-p    Iris Xe (Xe-LP)
//   lnl      Arc 140V / Lunar Lake (Xe2)
// A JIT image (AOT=false) runs on any card but pays the runtime JIT on the
// first kernel.
//
// Registry: set REGISTRY to the registry host (and optional path), e.g.
//   REGISTRY=ghcr.io/acme  IMAGE_NAMESPACE=sycl-infer
// produces ghcr.io/acme/sycl-infer/{all,worker,router}:<tag>.
// Leave REGISTRY empty for Docker Hub and set IMAGE_NAMESPACE to the org.
//
// Tags: the primary tag follows ci/Jenkinsfile's wheel version rule,
//   <python/VERSION>+<7-char hash>[.aot.<device>]
// with the device sanitized (acm-g10 -> acm.g10).  Docker forbids '+', so the
// tag is the same string with '+' written '-':
//   0.1.0-4fc5f29[.aot.acm.g10]
// The router is never AOT-built, so it never carries the suffix.  EXTRA_TAGS
// adds aliases (e.g. latest); IMAGE_TAG overrides the derived tag.
//
// The Dockerfiles accept BUILD_JOBS (the number of parallel compile/link
// jobs).  Leave it empty to use $(nproc) inside the container.  Lower it on a
// memory-constrained agent: a Debug SYCL build is OOM-killed at high -j, and
// the AOT link (ocloc) scales its RAM with the number of parallel jobs.
pipeline {
    agent any

    environment {
        // Needed for --platform and for the inline label/tag handling below.
        DOCKER_BUILDKIT = '1'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        // AOT lowering is the expensive step: one image per kernel, lowered by
        // ocloc at the final link.  The combined image lowers both binaries.
        timeout(time: 12, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    parameters {
        choice(name: 'IMAGE_TARGET',
               choices: ['all', 'worker', 'router', 'worker+router'],
               description: 'Which image(s) to build. "all" is the combined router+worker image.')

        booleanParam(name: 'AOT', defaultValue: true,
                     description: 'AOT-compile the device image instead of runtime JIT (worker/all only).')
        string(name: 'AOT_DEVICE', defaultValue: 'acm-g10',
               description: 'ocloc target when AOT=true: acm-g10 (Arc A770), adl-p (Iris Xe), lnl (Arc 140V / Lunar Lake).')
        string(name: 'ONEAPI_VERSION', defaultValue: '2026.2.0',
               description: 'oneAPI toolkit version. 2026.1.1 is ~16% faster for the 35B prefill (see AGENTS.md).')
        string(name: 'UBUNTU_TAG', defaultValue: 'ubuntu26.04',
               description: 'Ubuntu tag of the oneAPI base images (must match the host glibc).')
        string(name: 'DNNL_VERSION', defaultValue: '2026.1',
               description: 'oneDNN version bundled by the selected oneAPI toolkit.')
        string(name: 'RUNTIME_IMAGE', defaultValue: 'intel/oneapi-runtime',
               description: 'Runtime base image; use intel/oneapi-toolkit with INSTALL_GPU=false if the slim runtime is unavailable.')
        booleanParam(name: 'INSTALL_GPU', defaultValue: true,
                     description: 'Install the Level Zero driver / OpenCL ICD from the Intel graphics PPA in the runtime stage.')
        string(name: 'UBUNTU_MIRROR', defaultValue: 'archive.ubuntu.com',
               description: 'Ubuntu mirror rewritten into the image sources at build time.')
        string(name: 'BUILD_JOBS', defaultValue: '',
               description: 'Parallel compile jobs inside the image; empty = nproc. Lower it on a low-memory agent.')

        string(name: 'IMAGE_NAMESPACE', defaultValue: 'sycl-infer',
               description: 'Image namespace/org. The image is <registry>/<namespace>/<target>.')
        string(name: 'REGISTRY', defaultValue: '',
               description: 'Registry host (and optional path), e.g. ghcr.io/acme. Empty = Docker Hub.')
        string(name: 'IMAGE_TAG', defaultValue: '',
               description: 'Tag override. Empty = derive it from python/VERSION + the 7-char git hash (+ .aot.<device> when AOT), the same rule as ci/Jenkinsfile. Docker forbids "+", so it is written "-" (0.1.0-<hash>[.aot.<device>]).')
        string(name: 'EXTRA_TAGS', defaultValue: '',
               description: 'Additional tags applied to every image, comma-separated (e.g. latest).')
        booleanParam(name: 'PUSH_IMAGE', defaultValue: false,
                     description: 'Push the built images (requires REGISTRY; see REGISTRY_CREDENTIALS_ID).')
        string(name: 'REGISTRY_CREDENTIALS_ID', defaultValue: '',
               description: 'Jenkins username/password credential for docker login. Empty = assume the agent is already logged in.')

        booleanParam(name: 'RUN_SMOKE_TEST', defaultValue: true,
                     description: 'Run each built image and check the binaries / the router has no SYCL dependency.')
        booleanParam(name: 'PULL_BASE_IMAGE', defaultValue: true,
                     description: 'Pass --pull so the base image is refreshed.')
        booleanParam(name: 'NO_CACHE', defaultValue: false,
                     description: 'Pass --no-cache (a clean rebuild, much slower).')
        string(name: 'PLATFORM', defaultValue: 'linux/amd64',
               description: 'Build platform; empty = the agent\'s platform. Intel GPU hosts are amd64.')
        string(name: 'HTTP_PROXY', defaultValue: '',
               description: 'Build-time proxy for apt/wget inside the image (never stored in the image).')
        string(name: 'HTTPS_PROXY', defaultValue: '',
               description: 'Build-time HTTPS proxy for apt/wget inside the image.')
        string(name: 'NO_PROXY', defaultValue: 'localhost,127.0.0.1,::1',
               description: 'Proxy bypass list for the build.')
        string(name: 'EXTRA_BUILD_ARGS', defaultValue: '',
               description: 'Additional docker build flags, whitespace-separated (values must not contain spaces).')
    }

    stages {
        stage('Prepare') {
            steps {
                checkout scm
                script {
                    // ---- validation -------------------------------------------------
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

                    def ns = (params.IMAGE_NAMESPACE ?: '').trim()
                    ns = ns.replaceAll('^/+', '').replaceAll('/+$', '')
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

                    // ---- resolved values --------------------------------------------
                    env.GIT_SHA = sh(
                        returnStdout: true,
                        script: 'git rev-parse --short=7 HEAD 2>/dev/null || echo unknown'
                    ).trim()

                    def targets
                    if (params.IMAGE_TARGET == 'worker+router') {
                        targets = ['worker', 'router']
                    } else {
                        targets = [params.IMAGE_TARGET]
                    }
                    env.BUILD_TARGETS = targets.join(',')

                    // ---- version / tag (same rule as ci/Jenkinsfile) ----------------
                    // ci/Jenkinsfile builds the wheel version as
                    //   <python/VERSION> + <7-char hash> [ + .aot.<device> ]
                    // with the device sanitized by replacing every non-alnum run
                    // with ".".  A Docker tag cannot contain '+', so the Docker
                    // tag is the same string with '+' written as '-':
                    //   raw     0.1.0+<hash>.aot.acm-g10
                    //   version 0.1.0+<hash>.aot.acm.g10   (device: non-alnum -> ".")
                    //   tag     0.1.0-<hash>.aot.acm.g10   (Docker: "+" -> "-")
                    // The router is a plain host binary and is never AOT-built, so
                    // it never carries the .aot.<device> suffix.  IMAGE_TAG
                    // overrides the tag for all targets; EXTRA_TAGS adds aliases
                    // (e.g. latest) without changing the derived tag.
                    def baseVersion = readFile('python/VERSION').trim()
                    def aotDevice = ((params.AOT_DEVICE ?: '').trim() ?: 'unknown')
                        .replaceAll('[^A-Za-z0-9]+', '.')
                    env.VERSION_BASE = "${baseVersion}+${env.GIT_SHA}"
                    env.VERSION_AOT_SUFFIX = params.AOT ? ".aot.${aotDevice}" : ''
                    env.IMAGE_TAG_OVERRIDE = tagOverride
                    env.EXTRA_TAGS = extraTags

                    def prefix = registry ? "${registry}/" : ''
                    env.IMAGE_PREFIX = "${prefix}${ns}"

                    def exampleVersion = env.VERSION_BASE + env.VERSION_AOT_SUFFIX
                    def exampleTag = tagOverride ? tagOverride : exampleVersion.replace('+', '-')

                    echo """
                        |sycl-infer Docker build
                        |  targets       : ${env.BUILD_TARGETS}
                        |  image prefix  : ${env.IMAGE_PREFIX}  (repo = <prefix>/<target>)
                        |  version rule  : ${baseVersion}+<hash>[.aot.<device>]  (ci/Jenkinsfile rule)
                        |  derived tag   : ${exampleTag}  (Docker: '+' -> '-'; router: no AOT suffix)
                        |  extra tags    : ${extraTags ?: '-'}
                        |  git           : ${env.GIT_SHA}
                        |  AOT           : ${params.AOT} (device ${aotDevice}; the router is never AOT)
                        |  oneAPI        : ${params.ONEAPI_VERSION} / ${params.UBUNTU_TAG}
                        |  oneDNN        : ${params.DNNL_VERSION}
                        |  runtime base  : ${params.RUNTIME_IMAGE} (GPU stack: ${params.INSTALL_GPU})
                        |  build jobs    : ${jobs ?: 'nproc'}
                        |  push          : ${params.PUSH_IMAGE}
                        |  proxy         : ${(params.HTTP_PROXY ?: params.HTTPS_PROXY) ? 'set' : 'not set'}
                        |""".stripMargin()

                    // Fail early with a clear message if the agent has no daemon.
                    sh 'docker version --format "{{.Server.Version}}" >/dev/null'
                }
            }
        }

        stage('Build images') {
            steps {
                script {
                    def targets = env.BUILD_TARGETS.split(',') as List
                    def registry = (params.REGISTRY ?: '').trim().replaceAll('/+$', '')
                    def ns = (params.IMAGE_NAMESPACE ?: '').trim().replaceAll('^/+', '').replaceAll('/+$', '')
                    def prefix = registry ? "${registry}/" : ''
                    def repoFor = { String t -> "${prefix}${ns}/${t}".toString() }

                    // Version/tag rule shared with ci/Jenkinsfile.  The router is
                    // a plain host binary and is never AOT-built, so it never
                    // carries the .aot.<device> suffix.
                    def versionFor = { String t ->
                        def suffix = (params.AOT && t != 'router')
                            ? (env.VERSION_AOT_SUFFIX ?: '')
                            : ''
                        return (env.VERSION_BASE + suffix).toString()
                    }
                    def tagFor = { String t ->
                        def override = (env.IMAGE_TAG_OVERRIDE ?: '').trim()
                        return override ? override : versionFor(t).replace('+', '-')
                    }
                    def extraTags = (env.EXTRA_TAGS ?: '').split(/[,\s]+/).findAll { it }

                    def created = sh(
                        returnStdout: true,
                        script: 'date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown'
                    ).trim()

                    def dockerfiles = [
                        'all'   : 'docker/Dockerfile.all',
                        'worker': 'docker/Dockerfile.worker',
                        'router': 'docker/Dockerfile.router'
                    ]

                    def shQuote = { Object s ->
                        "'" + s.toString().replace("'", "'\\''") + "'"
                    }

                    targets.each { String target ->
                        def dockerfile = dockerfiles[target]
                        if (!dockerfile) {
                            error("Unknown IMAGE_TARGET '${target}'")
                        }

                        def repo = repoFor(target)
                        def tag = tagFor(target)
                        def tags = ([tag] + extraTags).unique()
                        def images = tags.collect { String t -> "${repo}:${t}".toString() }

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
                        args += ['-f', dockerfile]

                        // The worker/all Dockerfiles take the oneAPI + AOT
                        // arguments; the router image is a plain host binary
                        // and must not be told about an ocloc target.
                        if (target != 'router') {
                            args += ['--build-arg', "ONEAPI_VERSION=${params.ONEAPI_VERSION}".toString()]
                            args += ['--build-arg', "UBUNTU_TAG=${params.UBUNTU_TAG}".toString()]
                            args += ['--build-arg', "DNNL_VERSION=${params.DNNL_VERSION}".toString()]
                            args += ['--build-arg', "RUNTIME_IMAGE=${params.RUNTIME_IMAGE}".toString()]
                            args += ['--build-arg', "AOT=${params.AOT ? 'ON' : 'OFF'}".toString()]
                            args += ['--build-arg', "AOT_DEVICE=${params.AOT_DEVICE}".toString()]
                            args += ['--build-arg', "INSTALL_GPU=${params.INSTALL_GPU ? '1' : '0'}".toString()]
                            args += ['--build-arg', "UBUNTU_MIRROR=${params.UBUNTU_MIRROR}".toString()]
                        }

                        // Every Dockerfile now accepts BUILD_JOBS; the router
                        // ignores the cost of the SYCL device pass but its own
                        // compile and ctest still benefit from the bound.
                        def jobs = (params.BUILD_JOBS ?: '').trim()
                        if (jobs) {
                            args += ['--build-arg', "BUILD_JOBS=${jobs}".toString()]
                        }

                        // Proxies are predefined Docker build args; they reach
                        // the worker Dockerfiles through their explicit ARG and
                        // the router's apt through the predefined set.
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

                        images.each { String img -> args += ['-t', img] }

                        // OCI labels (no host names or private paths).
                        args += ['--label', "org.opencontainers.image.revision=${env.GIT_SHA}".toString()]
                        args += ['--label', "org.opencontainers.image.version=${versionFor(target)}".toString()]
                        args += ['--label', "org.opencontainers.image.created=${created}".toString()]

                        def extra = (params.EXTRA_BUILD_ARGS ?: '').trim()
                        if (extra) {
                            extra.split(/\s+/).each { String a -> args << a }
                        }

                        args << '.'

                        echo "Building ${target}: ${images.join(', ')}"
                        sh(script: args.collect { shQuote(it) }.join(' '),
                           label: "docker build ${target}")
                    }
                }
            }
        }

        stage('Smoke test') {
            when {
                expression { return params.RUN_SMOKE_TEST }
            }
            steps {
                script {
                    def targets = env.BUILD_TARGETS.split(',') as List
                    def registry = (params.REGISTRY ?: '').trim().replaceAll('/+$', '')
                    def ns = (params.IMAGE_NAMESPACE ?: '').trim().replaceAll('^/+', '').replaceAll('/+$', '')
                    def prefix = registry ? "${registry}/" : ''
                    def repoFor = { String t -> "${prefix}${ns}/${t}".toString() }
                    def versionFor = { String t ->
                        def suffix = (params.AOT && t != 'router')
                            ? (env.VERSION_AOT_SUFFIX ?: '')
                            : ''
                        return (env.VERSION_BASE + suffix).toString()
                    }
                    def tagFor = { String t ->
                        def override = (env.IMAGE_TAG_OVERRIDE ?: '').trim()
                        return override ? override : versionFor(t).replace('+', '-')
                    }

                    def shQuote = { Object s ->
                        "'" + s.toString().replace("'", "'\\''") + "'"
                    }

                    targets.each { String target ->
                        def img = "${repoFor(target)}:${tagFor(target)}".toString()
                        echo "Smoke-testing ${img}"

                        // --help parses and exits before any SYCL device query,
                        // so it runs on an agent without a GPU.
                        sh "docker run --rm ${shQuote(img)} --help >/dev/null"

                        if (target == 'all') {
                            sh "docker run --rm --entrypoint sh ${shQuote(img)} -c " +
                               shQuote('test -x /usr/local/bin/sycl-infer && ' +
                                        'test -x /usr/local/bin/sycl-infer-router && ' +
                                        'test -x /usr/local/bin/sycl-infer-onnx')
                            // The combined image's entrypoint is the router, so
                            // exercise the other two binaries explicitly: this is
                            // what proves the oneAPI runtime is complete.
                            sh "docker run --rm --entrypoint /usr/local/bin/sycl-infer ${shQuote(img)} --help >/dev/null"
                            sh "docker run --rm --entrypoint /usr/local/bin/sycl-infer-onnx ${shQuote(img)} --help >/dev/null"
                        }
                        if (target == 'worker') {
                            sh "docker run --rm --entrypoint sh ${shQuote(img)} -c " +
                               shQuote('test -x /usr/local/bin/sycl-infer')
                        }
                        if (target == 'router' || target == 'all') {
                            // The router must link neither SYCL nor oneDNN.
                            sh "! docker run --rm --entrypoint sh ${shQuote(img)} -c " +
                               shQuote('ldd /usr/local/bin/sycl-infer-router 2>/dev/null | grep -qi sycl')
                        }
                    }
                }
            }
        }

        stage('Push images') {
            when {
                expression { return params.PUSH_IMAGE }
            }
            steps {
                script {
                    def targets = env.BUILD_TARGETS.split(',') as List
                    def registry = (params.REGISTRY ?: '').trim().replaceAll('/+$', '')
                    if (!registry) {
                        error('PUSH_IMAGE=true requires REGISTRY')
                    }
                    def ns = (params.IMAGE_NAMESPACE ?: '').trim().replaceAll('^/+', '').replaceAll('/+$', '')
                    def prefix = "${registry}/"
                    def repoFor = { String t -> "${prefix}${ns}/${t}".toString() }
                    def versionFor = { String t ->
                        def suffix = (params.AOT && t != 'router')
                            ? (env.VERSION_AOT_SUFFIX ?: '')
                            : ''
                        return (env.VERSION_BASE + suffix).toString()
                    }
                    def tagFor = { String t ->
                        def override = (env.IMAGE_TAG_OVERRIDE ?: '').trim()
                        return override ? override : versionFor(t).replace('+', '-')
                    }
                    def extraTags = (env.EXTRA_TAGS ?: '').split(/[,\s]+/).findAll { it }

                    def images = []
                    targets.each { String t ->
                        def tags = ([tagFor(t)] + extraTags).unique()
                        images += tags.collect { String tg -> "${repoFor(t)}:${tg}".toString() }
                    }

                    def shQuote = { Object s ->
                        "'" + s.toString().replace("'", "'\\''") + "'"
                    }

                    // Login to the registry's host part (the first path element).
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

                    images.each { String img ->
                        sh "docker push ${shQuote(img)}"
                    }
                }
            }
        }
    }

    post {
        success {
            echo "Built ${env.BUILD_TARGETS} from ${env.GIT_SHA} (version ${env.VERSION_BASE}${env.VERSION_AOT_SUFFIX})."
        }
        always {
            echo 'Docker build finished; inspect the stage logs for the exact build command and tag list.'
        }
    }
}
