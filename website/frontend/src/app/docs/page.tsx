'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { 
  DocumentTextIcon,
  CpuChipIcon,
  CameraIcon,
  CubeIcon,
  BeakerIcon,
  ChartBarIcon,
  CodeBracketIcon,
  AcademicCapIcon,
  ArrowRightIcon,
  ArrowLeftIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'

const DocsPage = () => {
  const shouldReduceMotion = useReducedMotion()

  const docSections = [
    {
      title: 'Getting Started',
      icon: DocumentTextIcon,
      color: 'dental',
      description: 'Quick setup and installation guides',
      docs: [
        { name: 'System Requirements', href: '/docs/requirements', time: '5 min read' },
        { name: 'Installation Guide', href: '/docs/installation', time: '10 min read' },
        { name: 'First Scan Tutorial', href: '/docs/first-scan', time: '15 min read' },
        { name: 'Hardware Setup', href: '/docs/hardware-setup', time: '20 min read' }
      ]
    },
    {
      title: 'Technical Specifications',
      icon: CpuChipIcon,
      color: 'mint',
      description: 'Detailed technical documentation',
      docs: [
        { name: '3D Reconstruction Pipeline', href: '/docs/3d-pipeline', time: '12 min read' },
        { name: 'AI Model Architecture', href: '/docs/ai-models', time: '18 min read' },
        { name: 'Performance Benchmarks', href: '/docs/performance', time: '8 min read' },
        { name: 'Hardware Compatibility', href: '/docs/hardware', time: '10 min read' }
      ]
    },
    {
      title: 'API Reference',
      icon: CodeBracketIcon,
      color: 'dental',
      description: 'Complete API documentation',
      docs: [
        { name: 'REST API Overview', href: '/docs/api/rest', time: '15 min read' },
        { name: 'Scanning Endpoints', href: '/docs/api/scanning', time: '20 min read' },
        { name: 'AI Analysis API', href: '/docs/api/ai', time: '25 min read' },
        { name: 'Export Functions', href: '/docs/api/export', time: '12 min read' }
      ]
    },
    {
      title: 'Clinical Workflows',
      icon: BeakerIcon,
      color: 'mint',
      description: 'Professional dental workflows',
      docs: [
        { name: 'Intraoral Scanning Protocol', href: '/docs/workflows/intraoral', time: '15 min read' },
        { name: 'Impression Scanning', href: '/docs/workflows/impression', time: '12 min read' },
        { name: 'Quality Assessment', href: '/docs/workflows/quality', time: '10 min read' },
        { name: 'CAD/CAM Integration', href: '/docs/workflows/cadcam', time: '18 min read' }
      ]
    },
    {
      title: 'Advanced Features',
      icon: ChartBarIcon,
      color: 'dental',
      description: 'Advanced functionality and customization',
      docs: [
        { name: 'Custom AI Training', href: '/docs/advanced/ai-training', time: '30 min read' },
        { name: 'Performance Optimization', href: '/docs/advanced/optimization', time: '25 min read' },
        { name: 'Multi-Device Setup', href: '/docs/advanced/multi-device', time: '20 min read' },
        { name: 'Enterprise Deployment', href: '/docs/advanced/enterprise', time: '35 min read' }
      ]
    },
    {
      title: 'Troubleshooting',
      icon: CameraIcon,
      color: 'mint',
      description: 'Common issues and solutions',
      docs: [
        { name: 'Common Issues', href: '/docs/troubleshooting/common', time: '8 min read' },
        { name: 'Hardware Problems', href: '/docs/troubleshooting/hardware', time: '12 min read' },
        { name: 'Performance Issues', href: '/docs/troubleshooting/performance', time: '10 min read' },
        { name: 'Error Codes Reference', href: '/docs/troubleshooting/errors', time: '15 min read' }
      ]
    }
  ]

  const quickStart = [
    { step: '1', title: 'Install Software', description: 'Download and install OpenDentalScan' },
    { step: '2', title: 'Connect Hardware', description: 'Set up your scanning device' },
    { step: '3', title: 'Calibrate System', description: 'Run initial calibration' },
    { step: '4', title: 'Start Scanning', description: 'Begin your first dental scan' }
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
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: 'easeOut' }
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-dental-50 via-white to-mint-50">
      {/* Header */}
      <section className="pt-24 pb-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.div variants={itemVariants} className="mb-8">
              <Link
                href="/"
                className="inline-flex items-center space-x-2 text-dental-600 hover:text-dental-700 font-medium transition-colors duration-200"
              >
                <ArrowLeftIcon className="w-4 h-4" />
                <span>Back to Home</span>
              </Link>
            </motion.div>
            
            <motion.div variants={itemVariants} className="text-center">
              <div className="inline-flex items-center space-x-2 bg-dental-100 text-dental-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
                <DocumentTextIcon className="w-4 h-4" />
                <span>Technical Documentation</span>
              </div>
              
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-neutral-900 mb-6">
                Complete{' '}
                <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                  Documentation
                </span>
              </h1>
              
              <p className="text-xl text-neutral-600 max-w-3xl mx-auto mb-8">
                Comprehensive guides, technical specifications, and API documentation 
                for OpenDentalScan professional dental scanning solution.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Quick Start Guide */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-12">
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">Quick Start Guide</h2>
              <p className="text-neutral-600">Get up and running in 4 simple steps</p>
            </motion.div>

            <div className="grid md:grid-cols-4 gap-8">
              {quickStart.map((item, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="text-center"
                >
                  <div className="w-16 h-16 bg-gradient-to-br from-dental-500 to-mint-500 rounded-full flex items-center justify-center text-white text-2xl font-bold mx-auto mb-4">
                    {item.step}
                  </div>
                  <h3 className="text-lg font-semibold text-neutral-900 mb-2">{item.title}</h3>
                  <p className="text-neutral-600 text-sm">{item.description}</p>
                </motion.div>
              ))}
            </div>

            <motion.div variants={itemVariants} className="text-center mt-12">
              <Link
                href="/docs/getting-started"
                className="inline-flex items-center space-x-2 bg-dental-600 hover:bg-dental-700 text-white px-8 py-4 rounded-xl font-semibold text-lg transition-colors duration-200 shadow-dental"
              >
                <span>Start Quick Setup</span>
                <ArrowRightIcon className="w-5 h-5" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Documentation Sections */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-16">
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                Complete Documentation Library
              </h2>
              <p className="text-neutral-600 max-w-2xl mx-auto">
                Everything you need to master OpenDentalScan, from basic setup to advanced customization
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {docSections.map((section, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="bg-white rounded-2xl p-8 border border-neutral-200 hover:border-dental-200 transition-all duration-300 hover:shadow-gentle"
                >
                  <div className={`w-14 h-14 rounded-xl mb-6 flex items-center justify-center ${
                    section.color === 'dental'
                      ? 'bg-gradient-to-br from-dental-500 to-dental-600'
                      : 'bg-gradient-to-br from-mint-500 to-mint-600'
                  } shadow-soft`}>
                    <section.icon className="w-7 h-7 text-white" />
                  </div>

                  <h3 className="text-xl font-bold text-neutral-900 mb-3">{section.title}</h3>
                  <p className="text-neutral-600 mb-6">{section.description}</p>

                  <div className="space-y-3">
                    {section.docs.map((doc, docIndex) => (
                      <Link
                        key={docIndex}
                        href={doc.href}
                        className="flex items-center justify-between p-3 rounded-lg hover:bg-neutral-50 transition-colors duration-200 group"
                      >
                        <div className="flex items-center space-x-3">
                          <CheckCircleIcon className="w-4 h-4 text-mint-500" />
                          <span className="text-neutral-700 group-hover:text-dental-600 font-medium">
                            {doc.name}
                          </span>
                        </div>
                        <span className="text-xs text-neutral-500">{doc.time}</span>
                      </Link>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Technical Resources */}
      <section className="py-16 bg-gradient-to-br from-neutral-900 to-dental-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4">Technical Resources</h2>
              <p className="text-neutral-300 max-w-2xl mx-auto">
                Deep dive into the technology powering OpenDentalScan
              </p>
            </motion.div>

            <div className="grid md:grid-cols-3 gap-8">
              <motion.div variants={itemVariants} className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-dental-400 to-dental-500 rounded-2xl mx-auto mb-6 flex items-center justify-center">
                  <CubeIcon className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-semibold mb-4">3D Reconstruction</h3>
                <p className="text-neutral-300 mb-6">
                  Advanced TSDF fusion algorithms and real-time mesh generation techniques
                </p>
                <Link
                  href="/docs/3d-pipeline"
                  className="text-dental-300 hover:text-dental-200 font-medium"
                >
                  Learn More →
                </Link>
              </motion.div>

              <motion.div variants={itemVariants} className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-mint-400 to-mint-500 rounded-2xl mx-auto mb-6 flex items-center justify-center">
                  <CpuChipIcon className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-semibold mb-4">AI Architecture</h3>
                <p className="text-neutral-300 mb-6">
                  Neural network models for tooth segmentation and pathology detection
                </p>
                <Link
                  href="/docs/ai-models"
                  className="text-mint-300 hover:text-mint-200 font-medium"
                >
                  Learn More →
                </Link>
              </motion.div>

              <motion.div variants={itemVariants} className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-dental-400 to-mint-400 rounded-2xl mx-auto mb-6 flex items-center justify-center">
                  <ChartBarIcon className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Performance</h3>
                <p className="text-neutral-300 mb-6">
                  Benchmarks, optimization guides, and system requirements
                </p>
                <Link
                  href="/docs/performance"
                  className="text-dental-300 hover:text-dental-200 font-medium"
                >
                  Learn More →
                </Link>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Support CTA */}
      <section className="py-16 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants}>
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                Need Additional Support?
              </h2>
              <p className="text-xl text-neutral-600 mb-8">
                Our technical team is here to help you get the most out of OpenDentalScan
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  href="/support"
                  className="bg-dental-600 hover:bg-dental-700 text-white px-8 py-4 rounded-xl font-semibold text-lg transition-colors duration-200 shadow-dental"
                >
                  Contact Support
                </Link>
                <Link
                  href="/training"
                  className="bg-white hover:bg-neutral-50 text-dental-600 px-8 py-4 rounded-xl font-semibold text-lg border-2 border-dental-200 hover:border-dental-300 transition-all duration-200"
                >
                  Training Programs
                </Link>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default DocsPage