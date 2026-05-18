# Piper Audio - AI Audio Creation Platform

A premium AI audio creation platform for professional speech generation, podcast creation, document narration, and audio workflows.

## Platform Overview

Piper Audio is a modern, creator-focused platform that transforms text, documents, articles, and stories into natural AI narration. Built with a premium design system inspired by leading SaaS products, every workflow delivers professional-quality audio with an elegant user experience.

## Technical Architecture

- **Backend:** FastAPI (Python)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Design System:** Premium light-mode SaaS aesthetic
- **Typography:** Fraunces (headings) + Plus Jakarta Sans (body)
- **Deployment:** Docker Compose

### Project Structure

```text
Piper Audio/
├── app.py                    # FastAPI root application
├── index.html                # Premium homepage
├── about.html                # About page
├── privacy-policy.html       # Privacy policy
├── terms.html                # Terms of use
├── assets/
│   └── styles.css           # Premium design system
├── tools/
│   ├── audio_proofreader/   # Writing review workflow
│   ├── blog_to_podcast/     # Podcast creation workflow
│   ├── pdf_to_audio/        # Document narration workflow
│   ├── Text-to-audio/       # Speech generation workflow
│   └── story_narration_generator/  # Story narration workflow
└── docker-compose.yml       # Container orchestration
```

## AI Audio Workflows

### 1. AI Audio Proofreader
**Purpose:** Hear your writing before you send it  
**Use Case:** Professional communication, content review, tone refinement  
**Location:** `tools/audio_proofreader/`

### 2. Blog to Podcast Converter
**Purpose:** Transform articles into engaging podcast episodes  
**Use Case:** Content repurposing, podcast production, audio storytelling  
**Location:** `tools/blog_to_podcast/`

### 3. PDF to Audio Converter
**Purpose:** Convert documents into natural narration  
**Use Case:** Accessibility, passive listening, document review  
**Location:** `tools/pdf_to_audio/`

### 4. Text to Audio Converter
**Purpose:** Generate high-quality speech from any text  
**Use Case:** Scripts, notes, quick voice generation  
**Location:** `tools/Text-to-audio/`

### 5. AI Story Narration Generator
**Purpose:** Create immersive audio stories  
**Use Case:** Storytelling, audiobook creation, narrative content  
**Location:** `tools/story_narration_generator/`

## Development

### Run Locally

```bash
cd "Adsense Tools Page"

# Install dependencies for all workflows
pip install -r tools/pdf_to_audio/requirements.txt
pip install -r tools/Text-to-audio/requirements.txt
pip install -r tools/story_narration_generator/requirements.txt
pip install -r tools/audio_proofreader/requirements.txt
pip install -r tools/blog_to_podcast/requirements.txt

# Start the development server
uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

Open `http://127.0.0.1:8010/` to view the platform.

### Run With Docker

```bash
cd "Adsense Tools Page"
docker compose up --build
```

The compose file reads runtime settings from each tool's `.env` file.

## Design System

### Premium Light Mode Aesthetic
- **Primary Color:** #3a6df0 (refined blue)
- **Accent Color:** #7ca8ff (soft blue)
- **Background:** Subtle gradients with radial overlays
- **Typography:** Editorial scale (2.5rem - 5.8rem headlines)
- **Spacing:** Generous whitespace (6rem section padding)
- **Shadows:** Multi-layer soft shadows
- **Animations:** Smooth, intentional motion

### Inspiration
Design quality inspired by:
- Sarvam AI
- ElevenLabs
- Linear
- Vercel
- Modern AI-native startups

## Recent Updates

### 2026 Premium Redesign
✅ Complete UI/UX transformation  
✅ Premium design system implementation  
✅ Enhanced hero section with interactive visualizations  
✅ Asymmetric workflow showcase  
✅ Professional testimonials and social proof  
✅ Refined typography and spacing  
✅ Sophisticated animations and interactions  
✅ Mobile-optimized responsive design  
✅ SEO-optimized content positioning  

**See `REDESIGN_SUMMARY.md` for complete details.**

## Adding New Workflows

Follow the established pattern:

1. Create a new folder inside `tools/`
2. Include workflow-specific Python, HTML, env, requirements, and Docker assets
3. Keep outputs in the workflow's own folder
4. Mount or register the workflow from `app.py`
5. Apply the premium design system for consistency

## Contributing

When adding features or workflows:
- Maintain the premium design language
- Follow the established spacing and typography system
- Use the color palette consistently
- Ensure mobile responsiveness
- Add smooth, intentional animations
- Keep the creator-focused positioning

## License

See project documentation for licensing details.

---

**Piper Audio** - Professional AI audio creation for modern creators.
