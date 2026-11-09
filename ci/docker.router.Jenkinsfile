// Jenkins pipeline: build the sycl-infer router image ONLY
// (docker/Dockerfile.router).
//
// The router is a plain host program: it links neither SYCL nor oneDNN and
// needs no GPU, no icpx and no oneAPI runtime.  Its image is therefore built
// with a stock Debian toolchain, and the build stage runs the router's
// hermetic tests (test_router_*) as part of `docker build`.
//
// Use ci/docker.worker.Jenkinsfile for a single worker image, and
// ci/docker.Jenkinsfile with IMAGE_TARGET=all for the combined
// "router spawns its workers" image.
//
// The agent only needs a Docker daemon.
//
// Tags: <python/VERSION>-<7-char hash> - ci/Jenkinsfile's wheel version rule
// with '+' written '-' (a Docker tag cannot contain '+').  The router is never
// AOT-built, so it has no .aot.<device> suffix.  IMAGE_TAG overrides it;
// EXTRA_TAGS adds aliases such as latest.
pipeline {
    agent any

    environment {
        DOCKER_BUILDKIT = '1'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 2, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '30'))
    }

    parameters {
        string(name: 'IMAGE_NAMESPACE', defaultValue: 'sycl-infer',
               description: 'Image namespace/org: <registry>/<namespace>/router.')
        string(name: 'REGISTRY', defaultValue: '',
               description: 'Registry host (and optional path), e.g. ghcr.io/acme. Empty = Docker Hub.')
        string(name: 'IMAGE_TAG', defaultValue: '',
               description: 'Tag override. Empty = derive it from python/VERSION + the 7-char git hash, the same rule as ci/Jenkinsfile. Docker forbids "+", so it is written "-" (0.1.0-<hash>).')
        string(name: 'EXTRA_TAGS', defaultValue: '',
               description: 'Additional tags, comma-separated (e.g. latest).')
        booleanParam(name: 'PUSH_IMAGE', defaultValue: false,
                     description: 'Push the built image (requires REGISTRY).')
        string(name: 'REGISTRY_CREDENTIALS_ID', defaultValue: '',
               description: 'Jenkins username/password credential for docker login. Empty = assume already logged in.')

        booleanParam(name: 'RUN_SMOKE_TEST', defaultValue: true,
                     description: 'Run the image (--help) and assert the router binary has no SYCL dependency.')
        booleanParam(name: 'PULL_BASE_IMAGE', defaultValue: true,
                     description: 'Pass --pull so the Debian base image is refreshed.')
        booleanParam(name: 'NO_CACHE', defaultValue: false,
                     description: 'Pass --no-cache (a clean rebuild).')
        string(name: 'PLATFORM', defaultValue: '',
               description: 'Build platform; empty = the agent\'s platform (the router is architecture-neutral).')
        string(name: 'BUILD_JOBS', defaultValue: '',
               description: 'Parallel compile jobs inside the image; empty = nproc.')
        string(name: 'HTTP_PROXY', defaultValue: '',
               description: 'Build-time proxy for apt inside the image.')
        string(name: 'HTTPS_PROXY', defaultValue: '',
               description: 'Build-time HTTPS proxy for apt inside the image.')
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
                    //   <python/VERSION> + <7-char hash>
                    // The router is never AOT-built, so there is no
                    // .aot.<device> suffix.  A Docker tag cannot contain '+',
                    // so it is written '-'.
                    def baseVersion = readFile('python/VERSION').trim()
                    def version = "${baseVersion}+${env.GIT_SHA}"
                    env.VERSION_BASE = version
                    env.VERSION_AOT_SUFFIX = ''
                    env.IMAGE_TAG_OVERRIDE = tagOverride
                    env.EXTRA_TAGS = extraTags
                    env.IMAGE_TAG_VALUE = tagOverride ? tagOverride : version.replace('+', '-')
                    env.IMAGE_PREFIX = "${registry ? registry + '/' : ''}${ns}"

                    echo """
                        |sycl-infer router image
                        |  image       : ${env.IMAGE_PREFIX}/router:${env.IMAGE_TAG_VALUE}
                        |  version     : ${version}  (ci/Jenkinsfile rule)
                        |  extra tags  : ${extraTags ?: '-'}
                        |  git         : ${env.GIT_SHA}
                        |  build jobs  : ${jobs ?: 'nproc'}
                        |  push        : ${params.PUSH_IMAGE}
                        |  proxy       : ${(params.HTTP_PROXY ?: params.HTTPS_PROXY) ? 'set' : 'not set'}
                        |""".stripMargin()

                    sh 'docker version --format "{{.Server.Version}}" >/dev/null'
                }
            }
        }

        stage('Build router image') {
            steps {
                script {
                    def tag = env.IMAGE_TAG_VALUE
                    def repo = "${env.IMAGE_PREFIX}/router".toString()
                    def version = env.VERSION_BASE

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
                    args += ['-f', 'docker/Dockerfile.router']

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
                       label: 'docker build router')
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
                    def img = "${env.IMAGE_PREFIX}/router:${env.IMAGE_TAG_VALUE}".toString()

                    sh "docker run --rm ${shQuote(img)} --help >/dev/null"
                    sh "docker run --rm --entrypoint sh ${shQuote(img)} -c " +
                       shQuote('test -x /usr/local/bin/sycl-infer-router')
                    // The defining property of the router image: no SYCL link.
                    sh "! docker run --rm --entrypoint sh ${shQuote(img)} -c " +
                       shQuote('ldd /usr/local/bin/sycl-infer-router 2>/dev/null | grep -qi sycl')
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

                    def repo = "${env.IMAGE_PREFIX}/router".toString()
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
            echo "Built router image ${env.IMAGE_TAG_VALUE} from ${env.GIT_SHA} (version ${env.VERSION_BASE})."
        }
    }
}
