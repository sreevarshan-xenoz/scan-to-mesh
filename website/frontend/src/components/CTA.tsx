'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { 
  ArrowDownTrayIcon,
  DocumentTextIcon,
  PlayIcon,
  CodeBracketIcon,
  AcademicCapIcon,
  ChatBubbleLeftRightIcon,
  CheckCircleIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline'

const CTA = () => {
  const shouldReduceMotion = useReducedMotion()

  const ctaOptions = [
    {
      title: 'Download & Install',
      description: 'Get started with our complete installation package',
      icon: ArrowDownTrayIcon,
      color: 'dental',
      action: 'Download Now',
      features: ['Complete installer', 'Hardware setup guide', 'Sample data included']
    },
    {
      title: 'View Documentation',
      description: 'Comprehensive guides and API references',
      icon: DocumentTextIcon,
      color: 'mint',
      action: 'Read Docs',
      features: ['Technical specs', 'API reference', 'Tutorials']
    },
    {
      title: 'Try Live Demo',
      description: 'Interactive demonstration of all features',
      icon: PlayIcon,
      color: 'dental',
      action: 'Launch Demo',
      features: ['Real-time scanning', 'AI analysis', 'Export workflow']
    },
    {
      title: 'Professional Support',
      description: 'Expert technical support and training',
      icon: CodeBracketIcon,
      color: 'mint',
      action: 'Contact Sales',
      features: ['Technical support', 'Training programs', 'Custom solutions']
    }
  ]

  const quickLinks = [
    { name: 'Hardware Requirements', href: '#hardware' },
    { name: 'Installation Guide', href: '#install' },
    { name: 'API Documentation', href: '#api' },
    { name: 'Technical Support', href: '#support' },
    { name: 'Training Programs', href: '#training' },
    { name: 'Contact Sales', href: '#contact' }
  ]

  const containerVariants = shouldReduceMotion ? {} : {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  }

  const itemVariants = shouldReduceMotion ? {} : {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: 'easeOut' }
    }
  }

  return (
    <section id="cta" className="section-padding bg-white">
      <div className="container-max">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          {/* Main CTA Header */}
          <motion.div variants={itemVariants} className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-neutral-900 mb-6">
              Ready to Transform{' '}
              <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                Dental Scanning?
              </span>
            </h2>
            
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed mb-8">
              Join the open-source revolution in dental technology. Professional-grade scanning 
              capabilities, complete transparency, and unlimited customization.
            </p>

            {/* Primary CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              <motion.a
                href="/download"
                className="flex items-center justify-center space-x-3 bg-dental-600 hover:bg-dental-700 text-white px-8 py-4 rounded-xl font-bold text-lg shadow-dental transition-all duration-300"
                whileHover={shouldReduceMotion ? {} : { scale: 1.02, y: -2 }}
                whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
              >
                <ArrowDownTrayIcon className="w-6 h-6" />
                <span>Download OpenDentalScan</span>
                <ArrowRightIcon className="w-5 h-5" />
              </motion.a>

              <motion.a
                href="/demo"
                className="flex items-center justify-center space-x-3 bg-white hover:bg-neutral-50 text-dental-600 px-8 py-4 rounded-xl font-bold text-lg border-2 border-dental-200 hover:border-dental-300 transition-all duration-300"
                whileHover={shouldReduceMotion ? {} : { scale: 1.02, y: -2 }}
                whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
              >
                <PlayIcon className="w-6 h-6" />
                <span>Try Live Demo</span>
              </motion.a>
            </div>

            {/* Trust Indicators */}
            <div className="flex flex-wrap justify-center items-center gap-6 text-sm text-neutral-600">
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="w-4 h-4 text-mint-500" />
                <span>Professional Grade</span>
              </div>
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="w-4 h-4 text-mint-500" />
                <span>Cost Effective</span>
              </div>
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="w-4 h-4 text-mint-500" />
                <span>Modern Technology</span>
              </div>
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="w-4 h-4 text-mint-500" />
                <span>Expert Support</span>
              </div>
            </div>
          </motion.div>

          {/* CTA Options Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
            {ctaOptions.map((option, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                className="group bg-gradient-to-br from-white to-neutral-50 rounded-2xl p-6 border border-neutral-200 hover:border-dental-200 transition-all duration-300 hover:shadow-gentle"
              >
                {/* Icon */}
                <div className={`w-14 h-14 rounded-xl mb-4 flex items-center justify-center ${
                  option.color === 'dental'
                    ? 'bg-gradient-to-br from-dental-500 to-dental-600'
                    : 'bg-gradient-to-br from-mint-500 to-mint-600'
                } shadow-soft group-hover:shadow-dental transition-shadow duration-300`}>
                  <option.icon className="w-7 h-7 text-white" />
                </div>

                {/* Content */}
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-bold text-neutral-900 mb-2 group-hover:text-dental-700 transition-colors duration-200">
                      {option.title}
                    </h3>
                    <p className="text-neutral-600 text-sm leading-relaxed">
                      {option.description}
                    </p>
                  </div>

                  {/* Features */}
                  <div className="space-y-2">
                    {option.features.map((feature, featureIndex) => (
                      <div key={featureIndex} className="flex items-center space-x-2">
                        <div className={`w-1.5 h-1.5 rounded-full ${
                          option.color === 'dental' ? 'bg-dental-400' : 'bg-mint-400'
                        }`} />
                        <span className="text-xs text-neutral-600">{feature}</span>
                      </div>
                    ))}
                  </div>

                  {/* Action Button */}
                  <motion.button
                    className={`w-full py-3 rounded-lg font-semibold text-sm transition-all duration-200 ${
                      option.color === 'dental'
                        ? 'bg-dental-50 text-dental-700 hover:bg-dental-100'
                        : 'bg-mint-50 text-mint-700 hover:bg-mint-100'
                    }`}
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    {option.action}
                  </motion.button>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Quick Links */}
          <motion.div variants={itemVariants} className="bg-gradient-to-br from-dental-50 to-mint-50 rounded-2xl p-8 border border-dental-100">
            <div className="text-center mb-8">
              <h3 className="text-2xl font-bold text-neutral-900 mb-4">
                Quick Access Links
              </h3>
              <p className="text-neutral-600">
                Jump directly to the information you need
              </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {quickLinks.map((link, index) => (
                <motion.a
                  key={index}
                  href={link.href}
                  className="flex items-center justify-center space-x-2 bg-white hover:bg-dental-50 text-neutral-700 hover:text-dental-700 px-4 py-3 rounded-lg font-medium text-sm transition-all duration-200 border border-neutral-200 hover:border-dental-200"
                  whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                  whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                >
                  <span className="text-center">{link.name}</span>
                </motion.a>
              ))}
            </div>
          </motion.div>

          {/* Community & Support */}
          <motion.div variants={itemVariants} className="mt-16 text-center">
            <div className="bg-gradient-to-r from-dental-600 to-mint-500 rounded-3xl p-8 text-white">
              <div className="max-w-4xl mx-auto">
                <h3 className="text-3xl font-bold mb-4">
                  Join Our Growing Community
                </h3>
                <p className="text-xl mb-8 opacity-90">
                  Connect with dental professionals, researchers, and developers building 
                  the future of open-source dental technology.
                </p>

                <div className="grid md:grid-cols-3 gap-8 mb-8">
                  <div className="text-center">
                    <div className="w-16 h-16 bg-white/20 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                      <AcademicCapIcon className="w-8 h-8" />
                    </div>
                    <h4 className="text-lg font-semibold mb-2">Learn & Grow</h4>
                    <p className="text-sm opacity-90">Comprehensive documentation and tutorials</p>
                  </div>

                  <div className="text-center">
                    <div className="w-16 h-16 bg-white/20 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                      <ChatBubbleLeftRightIcon className="w-8 h-8" />
                    </div>
                    <h4 className="text-lg font-semibold mb-2">Get Support</h4>
                    <p className="text-sm opacity-90">Active community forum and help channels</p>
                  </div>

                  <div className="text-center">
                    <div className="w-16 h-16 bg-white/20 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                      <CodeBracketIcon className="w-8 h-8" />
                    </div>
                    <h4 className="text-lg font-semibold mb-2">Contribute</h4>
                    <p className="text-sm opacity-90">Help improve the platform for everyone</p>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <motion.button
                    className="bg-white text-dental-600 px-6 py-3 rounded-xl font-semibold hover:bg-neutral-50 transition-colors duration-200"
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    Join Community Forum
                  </motion.button>
                  
                  <motion.button
                    className="bg-transparent border-2 border-white text-white px-6 py-3 rounded-xl font-semibold hover:bg-white/10 transition-colors duration-200"
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    View on GitHub
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

export default CTA