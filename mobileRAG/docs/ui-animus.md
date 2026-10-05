# Animus UI

The app's visual language, 2026-09-26. The interface reads as part of the world it depicts,
not a skin laid over it.

## The fiction

The Hebbian memory is an entity born from your conversations. Every screen floats inside its
world instead of on a background.

- **Dark: the sea (awake).** Teal-black water lit by cyan "memory light".
- **Light: the white room.** Bright fog with a floor of cubes; this is where the
  phase-7 wake-up lives.

Home names the entity's state (AWAKE, or CONSOLIDATING while cortical consolidation runs), its
age (DAY n, counted from the first concept), and draws the memory itself as a constellation.

## Principles

1. **The UI lives inside the world.**
   - The atmosphere behind every screen: glow, drifting motes, a perspective grid floor,
     scanlines, and a vignette. In light mode: fog banks, towers, cubes, and rising motes.
   - Panels are frosted glass over it.
2. **One accent, used sparingly.** Vermilion, gold or cyan (Settings → Appearance).
   - In the sea it only marks things: the tick on each frame, "fading", the hub ring.
   - In the white room it is the selection slab.
   - The world light (cyan) does the rest.
3. **Type and position, not boxes.**
   - Barlow Semi Condensed: light and tracked out for titles, lowercase for menus.
   - Barlow for body text; IBM Plex Mono for data.
   - All three are bundled (OFL, `third_party/font-licenses/`), with no network use.
4. **Hard edges.** 2–4dp corners, no Material shadows, a 1px rim plus a top hairline.
5. **Craft details.**
   - Registration ticks at frame corners.
   - A red/cyan chromatic split on titles that occasionally flickers.
   - A scan line sweeping the entity panel.
   - A selection bar that fades out to the right and slides to the last entry opened.

## Where it lives

| File | What |
|---|---|
| `ui/theme/Color.kt` | Sea and fog palettes; accents (`mark` on dark, `slab` on light) |
| `ui/theme/Theme.kt` | Tokens (`GlassTokens`), fonts, shapes, typography |
| `ui/theme/Atmosphere.kt` | `AnimusBackground` (blur source + shared ambient clock), sea and white-room atmospheres, motion gating |
| `ui/theme/Animus.kt` | `frost`, `rimLight`, `cornerTicks`, `ChromaText`, `ArcGauge`, `DayBadge`, `HudStat`, `scanSweep`, `AnimusPanel` |
| `ui/theme/Glass.kt` | The older shared components, restyled, so every screen inherits the look |
| `hebbian/ui/Constellation.kt` | Graph → seeded force layout (`ConstellationLayout`, unit-tested) → glowing drawing |
| `home/HomeScreen.kt`, `hebbian/ui/HebbianGraphScreen.kt` | Custom-built from the mockups |
| `graph/ui/GraphLayout3D.kt`, `graph/ui/EntityConstellation3D.kt` | The entity graph as a 3D constellation (Graph → constellation tab): seeded 3D force layout with community cohesion and degree-normalised springs (tuned on the real graph via the opt-in `GraphLayoutRealDataTest`), perspective with depth fog, community hues, orbit rings, drag / pinch / double-tap reset / tap-to-inspect with the community summary |
| `hebbian/ui/TutorScreen.kt` | The tutor chat as a transmission log: frosted replies, your lines as the bright bar, an ARM-E meter strip and "remembered" concepts under each extraction turn, a diamond send |

## Tab bar

Drawn glyphs match the HUD: the entity (a ring holding a diamond), memory (a small
constellation), options (three sliders). The assistant and tutor tabs keep their avatars.

## Data honesty

Home's numbers come from the real graph.

- **Gauges:**
  - *linked*: concepts with any link.
  - *fading*: recall probability below ½.
  - *recalled*: concepts met at least twice.
- **HUD fading count:** before this change it was the size of a 5-row practice query, so it
  could never exceed 5.
- **Constellation:** the 14 concepts with the most link weight. Link brightness is scaled over
  the range of weights shown, because young graphs cluster near their starting weights.

## Performance (RedMagic 10 Pro, 90 Hz)

- Ambient motion stops when the screen isn't resumed, in battery saver, when animations are off
  in system settings, and while a list is scrolling (`PauseAmbientWhile`).
- One shared ambient clock publishes about 30 times a second. Separate clocks drifted apart and
  doubled the frame rate.
- The static atmosphere (water, light, scanlines, vignette; or sky, towers, cubes, fog) is baked
  once into one bitmap. As separate layers it was five full-screen fills per frame. The
  scanlines alone had been ~900 rasterised rects.
- **Frosted blur** (Haze 1.6.10, RenderEffect on API 31+, tinted fallback below):
  - Blurs a one-third-size copy.
  - Opt-in per panel (`glassPanel(frost = true)`); list cards use a plain tint.

| Home | First build | Now |
|---|---|---|
| Idle | ~64 fps, 11–13 ms GPU/frame | 30 fps, ~9 ms GPU/frame |
| Scrolling, missed frames | ~19% | 7–10% |

Idle GPU time per second is down about 65%.

**Constellation (97 entities, 477 links):** 30 fps, ~4 ms GPU, 1.3% missed frames. Labels are
measured once and faded with draw alpha; re-measuring each frame had cost 72% missed frames.

**Measured and rejected:**
- A smaller blur copy (⅕) and dropping the blur on moving panels: no gain.
- Per-layer bisection found the cost spread over the full-screen fills, hence the baking.

## Not done yet

- Chat (Khepri), Practice and Brain inherit the restyle but aren't custom-built yet; Graph's list view too (its constellation tab is custom).
- The Waking screen waits for phase 7; its mockup is on the canvas.
