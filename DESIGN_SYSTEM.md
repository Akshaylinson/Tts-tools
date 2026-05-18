# Piper Audio - Design System Guide

## Brand Identity

### Name
**Piper Audio**

### Tagline
"Professional AI audio creation for modern creators"

### Positioning
Premium AI audio creation platform (NOT a utility collection or tool hub)

---

## Color Palette

### Primary Colors
```css
--site-brand: #3a6df0        /* Primary blue - CTAs, links, accents */
--site-brand-deep: #234fca   /* Hover states, emphasis */
--site-accent: #7ca8ff       /* Secondary blue - highlights */
--site-accent-soft: #edf4ff  /* Light blue - backgrounds */
```

### Neutral Colors
```css
--site-ink: #142033          /* Primary text */
--site-ink-soft: #5f6f86     /* Secondary text */
--site-bg: #f6f7fb           /* Page background */
--site-panel: #ffffff        /* Card backgrounds */
--site-line: #dbe4f0         /* Borders */
```

### Gradients
```css
/* Primary Button */
background: linear-gradient(135deg, #3a6df0, #5b88ff);

/* Accent Button */
background: linear-gradient(135deg, #edf4ff, #dbe8ff);

/* Hero Background */
background: linear-gradient(180deg, #fbfcff 0%, #f4f6fb 48%, #eef2f8 100%);
```

---

## Typography

### Font Families
```css
/* Headlines */
font-family: 'Fraunces', Georgia, serif;

/* Body Text */
font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif;
```

### Type Scale
```css
/* Hero Title */
font-size: clamp(2.75rem, 7vw, 5.8rem);
line-height: 0.96;
letter-spacing: -0.045em;
font-weight: 800;

/* Section Heading */
font-size: clamp(2.5rem, 5vw, 4.2rem);
line-height: 1.05;
letter-spacing: -0.04em;
font-weight: 800;

/* Card Title */
font-size: 2rem - 3xl;
line-height: 1.2;
font-weight: 700;

/* Body Large */
font-size: 1.15rem;
line-height: 1.7;

/* Body Regular */
font-size: 1rem;
line-height: 1.7;

/* Small Text */
font-size: 0.9rem;
line-height: 1.6;
```

---

## Spacing System

### Section Padding
```css
/* Desktop */
padding: 6rem 0;

/* Mobile */
padding: 3rem 0;
```

### Component Padding
```css
/* Large Cards */
padding: 2.5rem;

/* Regular Cards */
padding: 2rem;

/* Compact Cards */
padding: 1.5rem;
```

### Gap System
```css
/* Large Gap */
gap: 2rem;

/* Medium Gap */
gap: 1.5rem;

/* Small Gap */
gap: 1rem;

/* Tight Gap */
gap: 0.75rem;
```

---

## Border Radius

### Component Radii
```css
/* Hero Cards, Major Sections */
border-radius: 2.5rem;

/* Standard Cards */
border-radius: 2rem;

/* Medium Components */
border-radius: 1.75rem;

/* Small Components */
border-radius: 1.25rem;

/* Buttons, Pills */
border-radius: 9999px;
```

---

## Shadows

### Shadow System
```css
/* Premium Card */
box-shadow: 0 32px 80px rgba(27, 45, 78, 0.12);

/* Standard Card */
box-shadow: 0 20px 56px rgba(27, 45, 78, 0.09);

/* Soft Card */
box-shadow: 0 16px 48px rgba(27, 45, 78, 0.08);

/* Subtle Card */
box-shadow: 0 12px 36px rgba(27, 45, 78, 0.06);

/* Button Shadow */
box-shadow: 0 16px 40px rgba(58, 109, 240, 0.24);
```

### Hover States
```css
/* Card Hover */
transform: translateY(-4px);
box-shadow: 0 24px 64px rgba(27, 45, 78, 0.14);

/* Button Hover */
transform: translateY(-2px);
box-shadow: 0 20px 48px rgba(58, 109, 240, 0.32);
```

---

## Component Patterns

### Primary Button
```html
<a href="#" class="premium-button">
  Start Free
</a>
```

```css
.premium-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 3.5rem;
  padding: 0 2rem;
  border-radius: 9999px;
  background: linear-gradient(135deg, var(--site-brand), #5b88ff);
  color: #fff;
  font-weight: 700;
  box-shadow: 0 16px 40px rgba(58, 109, 240, 0.24);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
```

### Secondary Button
```html
<a href="#" class="secondary-button">
  Learn More
</a>
```

```css
.secondary-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 3.5rem;
  padding: 0 2rem;
  border-radius: 9999px;
  background: rgba(255, 255, 255, 0.9);
  color: var(--site-ink);
  font-weight: 700;
  border: 1px solid var(--site-line);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
```

### Premium Card
```html
<article class="premium-card">
  <h3>Card Title</h3>
  <p>Card description text</p>
</article>
```

```css
.premium-card {
  padding: 2.5rem;
  border: 1px solid var(--site-line);
  border-radius: 2rem;
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 16px 48px rgba(27, 45, 78, 0.08);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.premium-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 24px 64px rgba(27, 45, 78, 0.14);
}
```

### Section Header
```html
<div class="section-intro">
  <span class="section-kicker">Section Label</span>
  <h2 class="section-heading">Main Heading</h2>
  <p class="section-copy">Supporting description text</p>
</div>
```

```css
.section-kicker {
  display: inline-flex;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--site-brand);
}

.section-heading {
  margin-top: 1rem;
  font-size: clamp(2.5rem, 5vw, 4.2rem);
  line-height: 1.05;
  font-weight: 800;
}

.section-copy {
  margin-top: 1rem;
  font-size: 1.1rem;
  line-height: 1.7;
  color: var(--site-ink-soft);
}
```

---

## Animation Guidelines

### Timing Functions
```css
/* Standard Ease */
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

/* Smooth Ease */
transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);

/* Snappy Ease */
transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
```

### Transform Patterns
```css
/* Hover Lift */
transform: translateY(-4px);

/* Button Lift */
transform: translateY(-2px);

/* Subtle Lift */
transform: translateY(-2px);
```

### Keyframe Animations
```css
/* Floating Orb */
@keyframes heroFloat {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(-30px, -40px, 0) scale(1.1); }
}

/* Waveform Pulse */
@keyframes wavePulse {
  0%, 100% { height: var(--height, 50%); opacity: 0.8; }
  50% { height: calc(var(--height, 50%) * 1.3); opacity: 1; }
}

/* Signal Sweep */
@keyframes signalSweep {
  to { transform: translateX(100%); }
}
```

---

## Layout Patterns

### Hero Layout
```html
<section class="home-hero">
  <div class="mx-auto max-w-7xl px-4 pb-20 pt-16">
    <div class="home-hero-grid">
      <div>
        <!-- Left: Content -->
      </div>
      <div>
        <!-- Right: Visual -->
      </div>
    </div>
  </div>
</section>
```

### Two-Column Grid
```css
.home-hero-grid {
  display: grid;
  gap: 4rem;
  align-items: center;
}

@media (min-width: 1024px) {
  .home-hero-grid {
    grid-template-columns: 1.1fr 0.9fr;
  }
}
```

### Card Grid
```css
.card-grid {
  display: grid;
  gap: 2rem;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}
```

---

## Responsive Breakpoints

### Mobile First Approach
```css
/* Mobile: Default styles */
.component {
  padding: 1.5rem;
  font-size: 1rem;
}

/* Tablet: 768px+ */
@media (min-width: 768px) {
  .component {
    padding: 2rem;
  }
}

/* Desktop: 1024px+ */
@media (min-width: 1024px) {
  .component {
    padding: 2.5rem;
    font-size: 1.1rem;
  }
}
```

### Key Breakpoints
- **Mobile:** < 768px
- **Tablet:** 768px - 1023px
- **Desktop:** 1024px+
- **Large Desktop:** 1280px+

---

## Accessibility

### Color Contrast
- Text on white: Minimum 4.5:1 ratio
- Large text: Minimum 3:1 ratio
- Interactive elements: Clear focus states

### Focus States
```css
button:focus,
a:focus {
  outline: 2px solid var(--site-brand);
  outline-offset: 2px;
}
```

### Motion Preferences
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## Content Guidelines

### Voice & Tone
- **Professional** but approachable
- **Clear** and concise
- **Creator-focused** language
- **Premium** positioning
- Avoid utility/tool language

### Headline Patterns
✅ "Create professional audio from any content"  
✅ "Transform documents into natural narration"  
✅ "Five powerful tools. One seamless experience"  

❌ "Useful tools for everyday tasks"  
❌ "Online utilities for content"  
❌ "Simple converters and generators"  

### CTA Patterns
✅ "Start Free"  
✅ "Explore Workflows"  
✅ "Create Podcast"  

❌ "Try Tool"  
❌ "Use Converter"  
❌ "Access Utility"  

---

## Icon System

### Emoji Icons (Current)
- 🎧 Audio/Listening
- 🎙️ Podcast/Recording
- 📝 Writing/Text
- 📖 Story/Narrative
- ⚡ Speed/Instant
- 🎯 Precision/Focus
- ✨ Quality/Premium

### Future: Custom Icon Set
Consider implementing custom SVG icons for:
- Consistency
- Scalability
- Brand uniqueness
- Animation control

---

## Performance Guidelines

### CSS Optimization
- Use `transform` and `opacity` for animations
- Avoid animating `width`, `height`, `top`, `left`
- Use `will-change` sparingly
- Minimize blur effects on mobile

### Loading Strategy
- Critical CSS inline
- Defer non-critical styles
- Lazy load below-fold images
- Optimize font loading

---

## Quality Checklist

### Before Launch
- [ ] All text uses correct font families
- [ ] Spacing follows the system (6rem sections, 2rem gaps)
- [ ] Shadows use approved values
- [ ] Border radius is consistent
- [ ] Hover states work smoothly
- [ ] Mobile layout is optimized
- [ ] Colors match the palette
- [ ] Typography scale is correct
- [ ] Animations are smooth (60fps)
- [ ] Focus states are visible
- [ ] Content avoids utility language
- [ ] SEO keywords are naturally integrated

---

## Design Inspiration

### Reference Platforms
Study these for quality benchmarks:
- **Sarvam AI** - Premium spacing, calm aesthetic
- **ElevenLabs** - Audio visualization, professional feel
- **Linear** - Clean layouts, smooth interactions
- **Vercel** - Typography, gradient usage
- **Stripe** - Documentation clarity

### Key Takeaways
- Generous whitespace = premium feel
- Subtle animations > flashy effects
- Editorial typography > generic fonts
- Soft shadows > hard borders
- Asymmetric layouts > rigid grids

---

**Piper Audio Design System v1.0**  
Last Updated: 2026  
Maintained by: Design Team
