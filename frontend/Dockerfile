FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json ./

# Install all dependencies with explicit ajv fix
RUN npm install --legacy-peer-deps --force && \
    npm install ajv@^8.12.0 --legacy-peer-deps --save-dev

# Copy source code
COPY . .

# Set NODE_OPTIONS to handle OpenSSL
ENV NODE_OPTIONS=--openssl-legacy-provider

# Build the application
RUN npm run build

# Production stage with nginx
FROM nginx:alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Copy built files from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
