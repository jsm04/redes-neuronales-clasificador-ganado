import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'Identificación de razas bovinas',
  description:
    'Clasificador de razas de ganado a partir de una fotografía, servido por una red EfficientNet-B0.',
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
