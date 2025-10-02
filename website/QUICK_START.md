# 🚀 Quick Start Guide

## Install Dependencies & Run

```bash
# Navigate to frontend directory
cd website/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Then visit: **http://localhost:3000**

## If You Get Errors

The TypeScript errors you're seeing are because dependencies aren't installed yet. Once you run `npm install`, all the errors will be resolved.

### Common Issues:

1. **"Cannot find module 'react'"** → Run `npm install`
2. **"Cannot find module 'framer-motion'"** → Run `npm install`
3. **JSX errors** → These will disappear after `npm install`

## What's Included

✅ **Professional dental website** with modern design  
✅ **Subtle animations** following dental industry standards  
✅ **Accessibility-first** with motion sensitivity support  
✅ **Mobile responsive** design  
✅ **TypeScript** for type safety  
✅ **Tailwind CSS** with custom dental color palette  

## Project Structure

```
website/frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx      # Root layout
│   │   ├── page.tsx        # Main page
│   │   └── globals.css     # Global styles
│   └── components/
│       ├── Navigation.tsx  # Header navigation
│       ├── Hero.tsx        # Hero section
│       ├── Features.tsx    # Features grid
│       ├── Demo.tsx        # Interactive demo
│       ├── TechSpecs.tsx   # Technical specs
│       ├── Comparison.tsx  # Market comparison
│       ├── CTA.tsx         # Call to action
│       └── Footer.tsx      # Footer
├── package.json            # Dependencies
├── tailwind.config.js      # Tailwind configuration
└── tsconfig.json          # TypeScript config
```

## Next Steps

1. **Install dependencies**: `npm install`
2. **Start dev server**: `npm run dev`
3. **Customize content**: Edit components in `src/components/`
4. **Deploy**: Use Vercel, Netlify, or Docker

## Need Help?

Check the full documentation in `PROJECT_OVERVIEW.md` for detailed information about the website architecture and customization options.