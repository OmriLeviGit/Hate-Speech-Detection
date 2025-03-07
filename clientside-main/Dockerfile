# frontend/Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files first
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Build the app
RUN npm run build

# Install serve to host the static files
RUN npm install -g serve

# Expose port (can be changed to 80 if needed)
EXPOSE 3000

# Run the app using serve
CMD ["serve", "-s", "build", "-l", "3000"]