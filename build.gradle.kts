plugins {
    id("application")
    id("java")
}

group = "net.atomica"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

tasks.test {
    useJUnitPlatform()
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    implementation(files("libs/slick-util.jar"))
    implementation(platform("org.lwjgl:lwjgl-bom:3.3.4"))
    implementation("org.lwjgl", "lwjgl")
    implementation("org.lwjgl", "lwjgl-assimp")
    implementation("org.lwjgl", "lwjgl-bgfx")
    implementation("org.lwjgl", "lwjgl-cuda")
    implementation("org.lwjgl", "lwjgl-egl")
    implementation("org.lwjgl", "lwjgl-fmod")
    implementation("org.lwjgl", "lwjgl-freetype")
    implementation("org.lwjgl", "lwjgl-glfw")
    implementation("org.lwjgl", "lwjgl-harfbuzz")
    implementation("org.lwjgl", "lwjgl-hwloc")
    implementation("org.lwjgl", "lwjgl-jawt")
    implementation("org.lwjgl", "lwjgl-jemalloc")
    implementation("org.lwjgl", "lwjgl-ktx")
    implementation("org.lwjgl", "lwjgl-libdivide")
    implementation("org.lwjgl", "lwjgl-llvm")
    implementation("org.lwjgl", "lwjgl-lmdb")
    implementation("org.lwjgl", "lwjgl-lz4")
    implementation("org.lwjgl", "lwjgl-meow")
    implementation("org.lwjgl", "lwjgl-meshoptimizer")
    implementation("org.lwjgl", "lwjgl-msdfgen")
    implementation("org.lwjgl", "lwjgl-nanovg")
    implementation("org.lwjgl", "lwjgl-nfd")
    implementation("org.lwjgl", "lwjgl-nuklear")
    implementation("org.lwjgl", "lwjgl-odbc")
    implementation("org.lwjgl", "lwjgl-openal")
    implementation("org.lwjgl", "lwjgl-opencl")
    implementation("org.lwjgl", "lwjgl-opengl")
    implementation("org.lwjgl", "lwjgl-opengles")
    implementation("org.lwjgl", "lwjgl-openvr")
    implementation("org.lwjgl", "lwjgl-openxr")
    implementation("org.lwjgl", "lwjgl-opus")
    implementation("org.lwjgl", "lwjgl-ovr")
    implementation("org.lwjgl", "lwjgl-par")
    implementation("org.lwjgl", "lwjgl-remotery")
    implementation("org.lwjgl", "lwjgl-rpmalloc")
    implementation("org.lwjgl", "lwjgl-shaderc")
    implementation("org.lwjgl", "lwjgl-spvc")
    implementation("org.lwjgl", "lwjgl-sse")
    implementation("org.lwjgl", "lwjgl-stb")
    implementation("org.lwjgl", "lwjgl-tinyexr")
    implementation("org.lwjgl", "lwjgl-tinyfd")
    implementation("org.lwjgl", "lwjgl-tootle")
    implementation("org.lwjgl", "lwjgl-vma")
    implementation("org.lwjgl", "lwjgl-vulkan")
    implementation("org.lwjgl", "lwjgl-xxhash")
    implementation("org.lwjgl", "lwjgl-yoga")
    implementation("org.lwjgl", "lwjgl-zstd")
    runtimeOnly("org.lwjgl", "lwjgl", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-assimp", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-bgfx", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-freetype", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-glfw", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-harfbuzz", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-hwloc", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-jemalloc", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-ktx", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-libdivide", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-llvm", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-lmdb", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-lz4", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-meow", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-meshoptimizer", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-msdfgen", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-nanovg", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-nfd", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-nuklear", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-openal", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-opengl", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-opengles", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-openvr", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-openxr", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-opus", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-ovr", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-par", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-remotery", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-rpmalloc", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-shaderc", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-spvc", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-sse", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-stb", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-tinyexr", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-tinyfd", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-tootle", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-vma", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-xxhash", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-yoga", classifier = "natives-windows")
    runtimeOnly("org.lwjgl", "lwjgl-zstd", classifier = "natives-windows")
    implementation("org.joml", "joml", "1.10.7")
    implementation("org.joml", "joml-primitives", "1.10.0")
    implementation("org.lwjglx", "lwjgl3-awt", "0.1.8")
}

sourceSets {
    main {
        java {
            srcDirs("src/main/java", "libs")
        }
    }
}

tasks.register<Copy>("copyNatives") {
    from(configurations.runtimeClasspath.get().filter { it.name.endsWith("natives-windows.jar") }.map { zipTree(it) })
    into("$buildDir/natives-windows")
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

tasks.register<JavaExec>("runJar") {
    dependsOn("build")
    group = "application"
    description = "Run the JAR with specified JVM arguments"
    classpath = files("$buildDir/libs/Atomica-1.0-SNAPSHOT.jar")
    mainClass.set("net.atomica.Main")
    jvmArgs = listOf("-Djava.library.path=$buildDir/natives-windows/x64/org/lwjgl")
}

tasks.register<Copy>("copySources") {
    from("src/main/java")
    into("$buildDir/classes/java/main")
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

tasks.build {
    dependsOn("copyNatives", "copySources")
}

tasks.jar {
    dependsOn("copyNatives", "copySources")
}