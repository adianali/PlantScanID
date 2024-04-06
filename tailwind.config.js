/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/*",
            "./node_modules/flowbite/**/*.js",
            "./static/src/**/*.js"
],
  theme: {
    extend: {},
  },
  plugins: [
    require('flowbite/plugin')
  ],
}
