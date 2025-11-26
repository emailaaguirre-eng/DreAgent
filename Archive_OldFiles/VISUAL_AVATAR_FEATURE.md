# Visual Avatar Feature 🐦

## Overview

Lea now has a **visual avatar** that appears and animates when she's speaking! This provides a clear visual indicator during voice conversations.

## Features

### ✨ Visual Representation
- **Animated Avatar**: A pulsing 🐦 (hummingbird) icon appears in the center of the chat display
- **Smooth Animation**: Opacity pulsing effect (fades in/out) while speaking
- **Glowing Border**: Green border with glow effect matching Lea's brand color (#68BD47)
- **Auto-Positioning**: Automatically centers in the chat area, even when window is resized

### 🎯 When It Appears
- **Shows**: When TTS (text-to-speech) starts playing Lea's response
- **Hides**: When TTS finishes or if there's an error
- **Works with**: Both edge-tts and gTTS engines

### 🎨 Visual Design
- **Size**: 120x120 pixels
- **Style**: Circular avatar with:
  - Green border (Lea's brand color)
  - Semi-transparent background
  - Large 🐦 emoji (80pt font)
  - Smooth pulsing opacity animation

## How It Works

1. **When Lea starts speaking:**
   - Avatar appears in center of chat
   - Pulsing animation begins
   - Green glow effect activates

2. **While speaking:**
   - Avatar pulses smoothly (opacity 0.5 to 1.0)
   - Animation loops continuously
   - Stays visible until speech ends

3. **When speech finishes:**
   - Animation stops
   - Avatar fades out and hides
   - Ready for next response

## Technical Details

### Implementation
- Uses PyQt6's `QPropertyAnimation` for smooth animations
- Overlay widget positioned on top of chat display
- Non-intrusive (doesn't block chat content)
- Thread-safe (updates on main UI thread)

### Performance
- Lightweight animation (minimal CPU usage)
- Smooth 60fps animation
- No impact on TTS playback

## Customization (Future)

Potential enhancements:
- Different avatar styles/emojis
- Custom animations (bouncing, rotating)
- Avatar images instead of emoji
- Size/position preferences
- Multiple animation styles

## Benefits

✅ **Visual Feedback**: Clear indication when Lea is speaking
✅ **Better UX**: Users know when TTS is active
✅ **Engaging**: Makes conversations feel more interactive
✅ **Accessibility**: Visual cue for hearing-impaired users
✅ **Professional**: Polished, modern interface

## Usage

The avatar appears automatically when:
1. TTS is enabled in Settings
2. Lea generates a response
3. TTS starts playing

No configuration needed - it just works! 🎉

