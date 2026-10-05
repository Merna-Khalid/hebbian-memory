import java.util.Properties

plugins {
    id("com.android.application")
    kotlin("android")
    kotlin("plugin.compose")
}

android {
    namespace = "com.mobilerag"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.mobilerag"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        // llama.cpp and LadybugDB are built for arm64 only; x86 slices were dead weight.
        ndk { abiFilters += "arm64-v8a" }
    }

    // Release signing: keystore/keystore.properties (kept out of version control). Every
    // update installed over a shared APK must be signed with this same key — back it up.
    val keystoreProps = Properties().apply {
        val f = rootProject.file("keystore/keystore.properties")
        if (f.isFile) f.inputStream().use { load(it) }
    }
    signingConfigs {
        if (keystoreProps.getProperty("storeFile") != null) {
            create("release") {
                storeFile = file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        getByName("release") {
            // No shrinking: LadybugDB, DJL tokenizers and ONNX Runtime reach classes by
            // reflection/JNI, and an obfuscated build isn't worth the risk here.
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release")
        }
        // A release build that installs beside the real app (".rc"), to test a clean first
        // run — the model download — without touching the installed app's data.
        create("rc") {
            initWith(getByName("release"))
            applicationIdSuffix = ".rc"
            versionNameSuffix = "-rc"
            matchingFallbacks += listOf("release")
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    // JVM unit tests exercise app classes that log via android.util.Log — return defaults
    // instead of throwing "Method ... not mocked".
    testOptions {
        unitTests.isReturnDefaultValues = true
    }

    // ONNX model files must not be compressed in the APK
    aaptOptions {
        noCompress("onnx")
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
        resources {
            excludes += setOf(
                "META-INF/DEPENDENCIES",
                "META-INF/LICENSE",
                "META-INF/LICENSE.txt",
                "META-INF/NOTICE",
                "META-INF/NOTICE.txt",
                "META-INF/*.SF",
                "META-INF/*.RSA",
                // The lbug jar's desktop natives (Linux/macOS/Windows) — the app uses the
                // prebuilt Android build under prebuilt/ladybug. JVM unit tests still get them
                // from the jar on the test classpath.
                "liblbug_java_native.so_*",
            )
        }
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.06.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.1")
    implementation("androidx.navigation:navigation-compose:2.9.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")

    // Frosted-glass backdrop blur for the Animus UI (RenderEffect on API 31+, tinted scrim below).
    implementation("dev.chrisbanes.haze:haze:1.6.10")

    // Phase 4: background community indexing (daily, device idle + charging)
    implementation("androidx.work:work-runtime-ktx:2.10.1")

    // Spike A: ONNX Runtime
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")

    // Spike B: LadybugDB Java API (note: jar ships no android_arm64 native lib — see Spike B)
    implementation("com.ladybugdb:lbug:0.20.3")

    // Spike C: llama.cpp (vendored from examples/llama.android)
    implementation(project(":llamacpp"))

    // HuggingFace fast tokenizer (tokenizer.json); tokenizer-native bundles libdjl_tokenizer.so for Android
    implementation("ai.djl.huggingface:tokenizers:0.33.0")
    implementation("ai.djl.android:tokenizer-native:0.33.0")

    // Step 6: ML Kit GenAI (Gemini Nano) availability probe
    implementation("com.google.mlkit:genai-prompt:1.0.0-beta4")
    implementation("com.google.mlkit:genai-common:1.0.0-beta4")

    // Tap-to-translate overlay in the tutor: on-device JA→EN. Model downloads once, then offline.
    implementation("com.google.mlkit:translate:17.0.3")

    // JVM unit tests for the pure-Kotlin Hebbian math (cortical GNN golden values, identity,
    // merge): run on the dev machine, no device. org.json is stubbed in android.jar, so the
    // golden-file reader needs the real one.
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
