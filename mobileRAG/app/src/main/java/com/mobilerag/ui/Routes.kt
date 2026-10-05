package com.mobilerag.ui

/** Navigation routes. Bottom-bar destinations are [Home], [Chat], [Tutor], [Memory], [Settings];
 *  the rest are reachable from the Home quick actions and the Settings hub. */
object Routes {
    const val Home = "home"
    const val Chat = "chat"
    const val Tutor = "tutor"
    const val Memory = "memory"
    const val Settings = "settings"

    const val Practice = "practice"
    const val Brain = "brain"
    const val Graph = "graph"
    const val Spikes = "spikes"
    const val Help = "help"

    const val SettingsAppearance = "settings/appearance"
    const val SettingsPerformance = "settings/performance"
    const val SettingsLearner = "settings/learner"
    const val SettingsSpaces = "settings/spaces"
    const val SettingsModels = "settings/models"
    const val SettingsData = "settings/data"
    const val SettingsAbout = "settings/about"

    val bottomBar = listOf(Home, Chat, Tutor, Memory, Settings)
}
