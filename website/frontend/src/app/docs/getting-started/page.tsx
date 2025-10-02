'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { 
  DocumentTextIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowLeftIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'

const GettingStartedPage = () => {
  const shouldReduceMotion = useReducedMotion()

  const requirements = [
    { category: 'Operating System', items: ['Windows 10/11 (64-bit)', 'macOS 12.0+', 'Ubuntu 20.04+'] },
    { category: 'Hardware', items: ['Intel i5-8400 / AMD Ryzen 5 2600+', '8GB RAM (16GB recommended)', 'NVIDIA GTX 1060+ / AMD RX 580+'] },
    { category: 'Storage', items: ['10GB free space', 'SSD recommended for performance'] },
    { category: 'Camera', items: ['Intel RealSense D435i/L515', 'USB 3.0 stereo cameras', 'Standard webcam (basic mode)'] }
  ]

  const installSteps = [
    {
      step: 1,
      title: 'Download OpenDentalScan',
      content: 'Download the latest version from our official website. Choose the installer for your operating system.',
      code: 'wget https://releases.opendentalscan.com/latest/OpenDentalScan-Setup.exe'
    },
    {
      step: 2,
      title: 'Run Installation',
      content: 'Execute the installer with administrator privileges. Follow the setup wizard to complete installation.',
      code: './OpenDentalScan-Setup.exe'
    },
    {
      step: 3,
      title: 'Connect Hardware',
      content: 'Connect your scanning device via USB 3.0. Ensure proper drivers are installed automatically.',
      code: null
    },
    {
      step: 4,
      title: 'Initial Configuration',
      content: 'Launch OpenDentalScan and run the initial setup wizard to configure your scanning hardware.',
      code: null
    },
    {
      step: 5,
      title: 'Calibration',
      content: 'Perform camera calibration using the provided calibration target for optimal accuracy.',
      code: null
    }
  ]

  const containerVariants = shouldReduceMotion ? undefined : {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  }

  const itemVariants = shouldReduceMotion ? undefined : {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.4, ease: 'easeOut' }
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-dental-50 via-white to-mint-50">
      {/* Header */}
      <section className="pt-24 pb-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.div variants={itemVariants}>
              <Link
                href="/docs"
                className="inline-flex items-center space-x-2 text-dental-600 hover:text-dental-700 font-medium mb-6 transition-colors duration-200"
              >
                <ArrowLeftIcon className="w-4 h-4" />
                <span>Back to Documentation</span>
              </Link>
              
              <div className="inline-flex items-center space-x-2 bg-dental-100 text-dental-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
                <DocumentTextIcon className="w-4 h-4" />
                <span>Getting Started Guide</span>
              </div>
              
              <h1 className="text-4xl md:text-5xl font-bold text-neutral-900 mb-6">
                Getting Started with{' '}
                <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                  OpenDentalScan
                </span>
              </h1>
              
              <p className="text-xl text-neutral-600 mb-8">
                Complete setup guide to get your OpenDentalScan system up and running in minutes.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Content */}
      <section className="pb-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-12"
          >
            {/* System Requirements */}
            <motion.div variants={itemVariants} className="bg-white rounded-2xl p-8 border border-neutral-200 shadow-soft">
              <h2 className="text-2xl font-bold text-neutral-900 mb-6 flex items-center space-x-3">
                <InformationCircleIcon className="w-6 h-6 text-dental-600" />
                <span>System Requirements</span>
              </h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                {requirements.map((req, index) => (
                  <div key={index} className="space-y-3">
                    <h3 className="font-semibold text-neutral-800">{req.category}</h3>
                    <ul className="space-y-2">
                      {req.items.map((item, itemIndex) => (
                        <li key={itemIndex} className="flex items-center space-x-2">
                          <CheckCircleIcon className="w-4 h-4 text-mint-500 flex-shrink-0" />
                          <span className="text-neutral-600 text-sm">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Installation Steps */}
            <motion.div variants={itemVariants} className="bg-white rounded-2xl p-8 border border-neutral-200 shadow-soft">
              <h2 className="text-2xl font-bold text-neutral-900 mb-6">Installation Steps</h2>
              
              <div className="space-y-8">
                {installSteps.map((step, index) => (
                  <div key={index} className="flex space-x-6">
                    <div className="flex-shrink-0">
                      <div className="w-10 h-10 bg-gradient-to-br from-dental-500 to-mint-500 rounded-full flex items-center justify-center text-white font-bold">
                        {step.step}
                      </div>
                    </div>
                    
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-neutral-900 mb-2">{step.title}</h3>
                      <p className="text-neutral-600 mb-4">{step.content}</p>
                      
                      {step.code && (
                        <div className="bg-neutral-900 rounded-lg p-4">
                          <code className="text-mint-300 font-mono text-sm">{step.code}</code>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Important Notes */}
            <motion.div variants={itemVariants} className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-2xl p-8 border border-yellow-200">
              <h2 className="text-2xl font-bold text-neutral-900 mb-6 flex items-center space-x-3">
                <ExclamationTriangleIcon className="w-6 h-6 text-yellow-600" />
                <span>Important Notes</span>
              </h2>
              
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-neutral-900">Hardware Compatibility</h3>
                    <p className="text-neutral-700">Ensure your scanning hardware is compatible before installation. Check our hardware compatibility guide for supported devices.</p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-neutral-900">GPU Drivers</h3>
                    <p className="text-neutral-700">Install the latest GPU drivers for optimal performance. NVIDIA drivers 470+ or AMD Adrenalin 21.10+ recommended.</p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-neutral-900">Firewall Settings</h3>
                    <p className="text-neutral-700">Configure firewall to allow OpenDentalScan network communication for license validation and updates.</p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Next Steps */}
            <motion.div variants={itemVariants} className="bg-gradient-to-br from-dental-600 to-mint-500 rounded-2xl p-8 text-white">
              <h2 className="text-2xl font-bold mb-6">Next Steps</h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                <Link
                  href="/docs/first-scan"
                  className="bg-white/20 backdrop-blur-sm rounded-xl p-6 hover:bg-white/30 transition-colors duration-200"
                >
                  <h3 className="font-semibold mb-2 flex items-center space-x-2">
                    <span>First Scan Tutorial</span>
                    <ArrowRightIcon className="w-4 h-4" />
                  </h3>
                  <p className="text-white/90 text-sm">Learn how to perform your first dental scan with step-by-step guidance.</p>
                </Link>
                
                <Link
                  href="/docs/hardware-setup"
                  className="bg-white/20 backdrop-blur-sm rounded-xl p-6 hover:bg-white/30 transition-colors duration-200"
                >
                  <h3 className="font-semibold mb-2 flex items-center space-x-2">
                    <span>Hardware Setup</span>
                    <ArrowRightIcon className="w-4 h-4" />
                  </h3>
                  <p className="text-white/90 text-sm">Detailed hardware configuration and calibration procedures.</p>
                </Link>
              </div>
            </motion.div>

            {/* Support */}
            <motion.div variants={itemVariants} className="text-center">
              <h2 className="text-2xl font-bold text-neutral-900 mb-4">Need Help?</h2>
              <p className="text-neutral-600 mb-6">
                Our support team is here to help you get started successfully.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  href="/support"
                  className="bg-dental-600 hover:bg-dental-700 text-white px-6 py-3 rounded-xl font-semibold transition-colors duration-200"
                >
                  Contact Support
                </Link>
                <Link
                  href="/docs/troubleshooting"
                  className="bg-white hover:bg-neutral-50 text-dental-600 px-6 py-3 rounded-xl font-semibold border-2 border-dental-200 hover:border-dental-300 transition-all duration-200"
                >
                  Troubleshooting Guide
                </Link>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default GettingStartedPage