/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  // react-pdf relies on browser-only APIs; opt the renderer out of SSR.
  experimental: {
    serverComponentsExternalPackages: ['pdfjs-dist', 'react-pdf'],
  },
};

export default nextConfig;
