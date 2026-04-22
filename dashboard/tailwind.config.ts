import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "hsl(217, 91%, 60%)",
        title: "hsl(244, 24%, 26%)",
        muted: "hsl(244, 16%, 43%)",
        surface: "hsl(258, 60%, 98%)",
      },
      fontFamily: {
        sans: ["Rubik", "sans-serif"],
      },
      borderRadius: {
        card: "20px",
      },
      boxShadow: {
        card: "0px 5px 20px 0px rgb(69 67 96 / 10%)",
      },
    },
  },
  plugins: [],
};

export default config;
