# OpenDentalScan Website

Modern, professional website for the OpenDentalScan project - an open-source dental scanning solution.

## 🚀 Features

- **Professional Design**: Clean, trustworthy design following dental industry standards
- **Subtle Animations**: Gentle, precise animations that inspire confidence
- **Accessibility First**: Respects `prefers-reduced-motion` and follows WCAG guidelines
- **Modern Tech Stack**: Next.js 14, TypeScript, Tailwind CSS, Framer Motion
- **3D Visualizations**: Interactive demos using React Three Fiber
- **Performance Optimized**: Fast loading, optimized images, efficient animations

## 🎨 Design Principles

Following dental industry best practices:
- **Trust & Care**: Professional color palette (blues, greens, whites)
- **Precision**: Clean typography and precise spacing
- **Gentle Motion**: Subtle animations (0.3s-0.6s duration)
- **Accessibility**: Motion-sensitive user support
- **Mobile-First**: Responsive design for all devices

## 🛠️ Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom dental color palette
- **Animations**: Framer Motion with reduced-motion support
- **3D Graphics**: React Three Fiber + Drei
- **Icons**: Heroicons
- **Deployment**: Vercel/Netlify ready

## 📦 Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## 🎯 Key Sections

1. **Hero**: Professional introduction with animated scanner interface
2. **Features**: Core capabilities with subtle hover effects
3. **Demo**: Interactive demonstration of scanning workflow
4. **Tech Specs**: Comprehensive technical specifications
5. **Comparison**: Open-source vs commercial solutions
6. **CTA**: Multiple engagement options and community links

## 🎨 Color Palette

### Dental (Primary Blue)
- `dental-50`: #f0f9ff (Very light - clean, sterile)
- `dental-500`: #0ea5e9 (Primary - confident)
- `dental-600`: #0284c7 (Deep - reliable)

### Mint (Secondary Green)
- `mint-50`: #f0fdf4 (Very light - fresh, clean)
- `mint-500`: #22c55e (Primary - growth)
- `mint-600`: #16a34a (Deep - stability)

### Neutral (Supporting Gray)
- `neutral-50`: #fafafa (Almost white - pure)
- `neutral-600`: #525252 (Dark - strong)
- `neutral-900`: #171717 (Near black - elegant)

## ⚡ Performance Features

- **Optimized Images**: Next.js Image component with lazy loading
- **Code Splitting**: Automatic route-based code splitting
- **Animation Optimization**: GPU-accelerated transforms
- **Reduced Motion**: Respects user accessibility preferences
- **Fast Loading**: Optimized bundle size and critical CSS

## 🔧 Customization

### Adding New Sections
1. Create component in `src/components/`
2. Add to main page in `src/app/page.tsx`
3. Follow existing animation patterns
4. Maintain dental color scheme

### Animation Guidelines
- Use `useReducedMotion()` hook for accessibility
- Keep durations between 0.3s-0.6s
- Prefer `ease-out` timing functions
- Use subtle transforms (scale: 1.02, y: -2px)

## 📱 Responsive Design

- **Mobile**: Optimized touch targets, simplified layouts
- **Tablet**: Balanced grid layouts, readable typography
- **Desktop**: Full feature showcase, hover interactions
- **Large Screens**: Proper max-widths, centered content

## 🚀 Deployment

### Vercel (Recommended)
```bash
npm install -g vercel
vercel
```

### Netlify
```bash
npm run build
# Upload dist folder to Netlify
```

### Docker
```bash
docker build -t opendentalscan-website .
docker run -p 3000:3000 opendentalscan-website
```

## 🤝 Contributing

1. Follow the design principles (trust, care, precision)
2. Maintain accessibility standards
3. Test with `prefers-reduced-motion`
4. Keep animations subtle and professional
5. Use the established color palette

## 📄 License

MIT License - see LICENSE file for details.

---

Built with ❤️ for the dental community