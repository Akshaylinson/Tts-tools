module.exports = {
  content: [
    "./index.html",
    "./about.html",
    "./privacy-policy.html",
    "./terms.html",
    "./tools/**/*.html"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#10261f",
        slatewarm: "#f8fafc",
        mist: "#f6f4eb",
        brand: "#166534",
        branddeep: "#0f3d2e",
        moss: "#d8e8cb",
        accent: "#f59e0b"
      },
      boxShadow: {
        card: "0 20px 55px rgba(16, 38, 31, 0.08)"
      }
    }
  },
  plugins: []
};
