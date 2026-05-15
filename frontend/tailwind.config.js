/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "node-running": "#3B82F6",
        "node-complete": "#22C55E",
        "node-failed": "#EF4444",
        "node-pending": "#6B7280",
        "node-hitl": "#A855F7",
      },
    },
  },
  plugins: [],
};
