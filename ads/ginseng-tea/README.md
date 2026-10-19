# Instant Ginseng Tea: UGC ad (FR)

| File | Length | Use |
|---|---|---|
| `ginseng-tea-ugc-20s.mp4` | 20.3 s | Feed ad (TikTok / Reels / Facebook) |
| `ginseng-tea-ugc-14s-stories.mp4` | 13.7 s | Stories / Reels cut-down |

Both are 1080x1920, 24 fps, H.264 + AAC, -14 LUFS integrated, true peak below -1.5 dBTP.

## Edit structure (20 s version)
- 0.0–1.5 s: hook. Tight face crop, punch-in to 110%, on-screen hook, caption "Après une longue journée".
- 1.5–9.2 s: problem → solution. Jump cuts alternating 100% / 112% framing.
- 9.2–13.2 s: product (Clip B). Box framed large, callouts: Instantané, Prêt en 2 min, 12 sachets.
- 13.2–15.2 s: emotion. Smile with the cup, then the box reveal. No dialogue; the music swells.
- 15.2–20.3 s: CTA. "Clique sur le bouton… pour commander. Paiement à la livraison." Animated "COMMANDER MAINTENANT" button with a bouncing arrow; the music lifts.

## Hooks for A/B testing (on-screen text, 0–2.5 s)
1. **Primary (used in the videos):** "Mon rituel du soir / en 2 minutes"
2. "Après une longue journée, tu fais quoi ?"
3. "Le thé que je me prépare chaque soir"
4. "Un vrai moment pour toi, prêt en 2 minutes"

All hooks avoid health, energy or cure claims.

## Rebuild
`tools/render.py <workdir> main|cut` expects `src/A.mp4`, `src/B.mp4` and cleaned `src/{A,B}_voice.wav` in `<workdir>`.
Music and sound effects are generated in code (`tools/audio.py`). No stock assets are used.
