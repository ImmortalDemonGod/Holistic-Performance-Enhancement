module.exports = {
  testEnvironment: 'jest-environment-jsdom',
  // If you have ES modules or specific syntax:
  transform: {
    '^.+\\.m?js$': 'babel-jest',
  },
  roots: ["<rootDir>/static"],
  testMatch: [
    "**/__tests__/**/*.js",
    "**/?(*.)+(spec|test).js"
  ],
  verbose: true,
};
