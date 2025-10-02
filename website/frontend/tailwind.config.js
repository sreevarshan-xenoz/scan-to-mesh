/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Professional dental color palette - trust, care, cleanliness
        dental: {
          50: '#f0f9ff',   // Very light blue - clean, sterile
          100: '#e0f2fe',  // Light blue - calming
          200: '#bae6fd',  // Soft blue - trustworthy
          300: '#7dd3fc',  // Medium blue - professional
          400: '#38bdf8',  // Bright blue - modern
          500: '#0ea5e9',  // Primary blue - confident
          600: '#0284c7',  // Deep blue - reliable
          700: '#0369a1',  // Darker blue - authoritative
          800: '#075985',  // Navy blue - premium
          900: '#0c4a6e',  // Dark navy - sophisticated
        },
        mint: {
          50: '#f0fdf4',   // Very light green - fresh, clean
          100: '#dcfce7',  // Light green - healthy
          200: '#bbf7d0',  // Soft green - natural
          300: '#86efac',  // Medium green - vibrant health
          400: '#4ade80',  // Bright green - vitality
          500: '#22c55e',  // Primary green - growth
          600: '#16a34a',  // Deep green - stability
          700: '#15803d',  // Forest green - trust
          800: '#166534',  // Dark green - premium
          900: '#14532d',  // Very dark green - luxury
        },
        neutral: {
          50: '#fafafa',   // Almost white - pure
          100: '#f5f5f5',  // Light gray - clean
          200: '#e5e5e5',  // Soft gray - subtle
          300: '#d4d4d4',  // Medium gray - balanced
          400: '#a3a3a3',  // Gray - professional
          500: '#737373',  // Mid gray - reliable
          600: '#525252',  // Dark gray - strong
          700: '#404040',  // Charcoal - premium
          800: '#262626',  // Very dark - sophisticated
          900: '#171717',  // Near black - elegant
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        // Subtle, professional animations - dentist's hand: precise, gentle, reassuring
        'fade-in': 'fadeIn 0.4s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'slide-in-right': 'slideInRight 0.4s ease-out',
        'scale-in': 'scaleIn 0.3s ease-out',
        'pulse-gentle': 'pulseGentle 2s ease-in-out infinite',
        'float-subtle': 'floatSubtle 4s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideInRight: {
          '0%': { transform: 'translateX(20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        pulseGentle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
        floatSubtle: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
        'gentle': '0 4px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        'dental': '0 8px 30px -12px rgba(2, 132, 199, 0.25)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}