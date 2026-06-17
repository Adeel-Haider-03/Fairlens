/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink:    { DEFAULT: '#0A0E1A', 50: '#f0f2f8', 100: '#e0e4f0', 200: '#b3bdda', 300: '#7b8db8', 400: '#4a5f96', 500: '#2a3a6e', 600: '#1a2550', 700: '#111a3a', 800: '#0D1228', 900: '#0A0E1A' },
        cyan:   { DEFAULT: '#00E5FF', 50: '#e0faff', 100: '#b3f4ff', 200: '#66eaff', 300: '#1adfff', 400: '#00E5FF', 500: '#00bcd4', 600: '#0097a7', 700: '#006978', 800: '#003d47', 900: '#001a1f' },
        amber:  { DEFAULT: '#FFB830', 50: '#fff8e1', 100: '#ffecb3', 200: '#ffe082', 300: '#ffd54f', 400: '#FFB830', 500: '#fb8c00', 600: '#e65100', 700: '#bf360c', },
        rose:   { DEFAULT: '#FF4D6D', 50: '#ffe4e9', 100: '#ffb3bf', 200: '#ff8096', 300: '#FF4D6D', 400: '#e0003a', 500: '#b0002e' },
        emerald:{ DEFAULT: '#00E676', 50: '#e0fff0', 100: '#b3ffd9', 200: '#66ffb3', 300: '#1aff8e', 400: '#00E676', 500: '#00c853', 600: '#009624' },
      },
      fontFamily: {
        display: ['"Space Mono"', 'monospace'],
        body:    ['"DM Sans"', 'sans-serif'],
        mono:    ['"Space Mono"', 'monospace'],
      },
      backgroundImage: {
        'grid-pattern': 'linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px)',
      },
      backgroundSize: {
        'grid': '40px 40px',
      }
    },
  },
  plugins: [],
}
