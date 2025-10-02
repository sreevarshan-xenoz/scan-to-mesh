import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'OpenDentalScan - Professional Dental Scanner',
  description: 'Professional-grade dental scanning solution built with modern open-source technologies. Real-time 3D reconstruction, AI-powered analysis, and clinical workflow integration.',
  keywords: 'dental scanner, 3D reconstruction, open source, intraoral scanning, dental technology',
  authors: [{ name: 'OpenDentalScan Team' }],
  openGraph: {
    title: 'OpenDentalScan - Professional Dental Scanner',
    description: 'Professional-grade dental scanning solution built with modern technology',
    type: 'website',
    images: ['/og-image.jpg'],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'OpenDentalScan - Professional Dental Scanner',
    description: 'Professional-grade dental scanning solution built with modern technology',
    images: ['/og-image.jpg'],
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} antialiased`}>
        {children}
      </body>
    </html>
  )
}
