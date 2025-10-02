'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { 
  ArrowTopRightOnSquareIcon,
  HeartIcon,
  CodeBracketIcon,
  AcademicCapIcon,
  ChatBubbleLeftRightIcon,
  DocumentTextIcon,
  BeakerIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline'

const Footer = () => {
  const shouldReduceMotion = useReducedMotion()

  const footerSections = [
    {
      title: 'Product',
      links: [
        { name: 'Features', href: '#features' },
        { name: 'Demo', href: '#demo' },
        { name: 'Technical Specs', href: '#specs' },
        { name: 'Comparison', href: '#comparison' },
        { name: 'Download', href: '#download' },
        { name: 'Roadmap', href: '#roadmap' }
      ]
    },
    {
      title: 'Documentation',
      links: [
        { name: 'Getting Started', href: '#docs/getting-started' },
        { name: 'API Reference', href: '#docs/api' },
        { name: 'Hardware Setup', href: '#docs/hardware' },
        { name: 'AI Models', href: '#docs/ai' },
        { name: 'Tutorials', href: '#docs/tutorials' },
        { name: 'FAQ', href: '#docs/faq' }
      ]
    },
    {
      title: 'Community',
      links: [
        { name: 'Product Updates', href: '#updates' },
        { name: 'Discussion Forum', href: '#community/forum' },
        { name: 'Discord Server', href: '#community/discord', external: true },
        { name: 'User Community', href: '#community/users' },
        { name: 'Best Practices', href: '#community/practices' },
        { name: 'Research Papers', href: '#research' }
      ]
    },
    {
      title: 'Support',
      links: [
        { name: 'Help Center', href: '#support' },
        { name: 'Bug Reports', href: '#support/bugs' },
        { name: 'Feature Requests', href: '#support/features' },
        { name: 'Professional Services', href: '#support/professional' },
        { name: 'Training', href: '#support/training' },
        { name: 'Contact Us', href: '#contact' }
      ]
    }
  ]

  const socialLinks = [
    { name: 'LinkedIn', href: '#linkedin', icon: CodeBracketIcon },
    { name: 'Research', href: '#research', icon: AcademicCapIcon },
    { name: 'Community', href: '#community', icon: ChatBubbleLeftRightIcon },
    { name: 'Documentation', href: '#docs', icon: DocumentTextIcon }
  ]

  const stats = [
    { label: 'Downloads', value: '10K+' },
    { label: 'Contributors', value: '50+' },
    { label: 'Research Papers', value: '15+' },
    { label: 'Countries', value: '25+' }
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
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.4, ease: 'easeOut' }
    }
  }

  return (
    <footer className="bg-gradient-to-br from-neutral-900 to-dental-900 text-white">
      <div className="container-max">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          {/* Main Footer Content */}
          <div className="section-padding border-b border-white/10">
            <div className="grid lg:grid-cols-5 gap-12">
              {/* Brand Section */}
              <motion.div variants={itemVariants} className="lg:col-span-1">
                <div className="flex items-center space-x-3 mb-6">
                  <div className="w-12 h-12 bg-gradient-to-br from-dental-400 to-mint-400 rounded-xl flex items-center justify-center shadow-dental">
                    <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">OpenDentalScan</h3>
                    <p className="text-sm text-neutral-300">Open Source</p>
                  </div>
                </div>
                
                <p className="text-neutral-300 leading-relaxed mb-6">
                  Professional-grade dental scanning made accessible through open-source innovation. 
                  Empowering dental professionals worldwide with transparent, customizable technology.
                </p>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-4">
                  {stats.map((stat, index) => (
                    <div key={index} className="text-center bg-white/5 rounded-lg p-3">
                      <div className="text-lg font-bold text-dental-300">{stat.value}</div>
                      <div className="text-xs text-neutral-400">{stat.label}</div>
                    </div>
                  ))}
                </div>
              </motion.div>

              {/* Footer Links */}
              {footerSections.map((section, sectionIndex) => (
                <motion.div key={sectionIndex} variants={itemVariants}>
                  <h4 className="text-lg font-semibold mb-6 text-dental-300">
                    {section.title}
                  </h4>
                  <ul className="space-y-3">
                    {section.links.map((link, linkIndex) => (
                      <li key={linkIndex}>
                        <motion.a
                          href={link.href}
                          className="flex items-center space-x-2 text-neutral-300 hover:text-white transition-colors duration-200 group"
                          target={link.external ? '_blank' : undefined}
                          rel={link.external ? 'noopener noreferrer' : undefined}
                          whileHover={shouldReduceMotion ? {} : { x: 4 }}
                          transition={{ duration: 0.2 }}
                        >
                          <span>{link.name}</span>
                          {link.external && (
                            <ArrowTopRightOnSquareIcon className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                          )}
                        </motion.a>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Newsletter & Social */}
          <motion.div variants={itemVariants} className="py-12 border-b border-white/10">
            <div className="grid md:grid-cols-2 gap-8 items-center">
              {/* Newsletter */}
              <div>
                <h4 className="text-2xl font-bold mb-4">Stay Updated</h4>
                <p className="text-neutral-300 mb-6">
                  Get the latest updates on new features, research, and community developments.
                </p>
                
                <div className="flex flex-col sm:flex-row gap-4">
                  <input
                    type="email"
                    placeholder="Enter your email"
                    className="flex-1 px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-neutral-400 focus:outline-none focus:border-dental-400 transition-colors duration-200"
                  />
                  <motion.button
                    className="bg-dental-600 hover:bg-dental-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors duration-200 shadow-dental"
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    Subscribe
                  </motion.button>
                </div>
              </div>

              {/* Social Links */}
              <div className="text-center md:text-right">
                <h4 className="text-xl font-semibold mb-6">Connect With Us</h4>
                <div className="flex justify-center md:justify-end space-x-4">
                  {socialLinks.map((social, index) => (
                    <motion.a
                      key={index}
                      href={social.href}
                      className="w-12 h-12 bg-white/10 hover:bg-dental-600 rounded-xl flex items-center justify-center transition-colors duration-200 group"
                      target={social.href.startsWith('http') ? '_blank' : undefined}
                      rel={social.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                      whileHover={shouldReduceMotion ? {} : { scale: 1.1, y: -2 }}
                      whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
                    >
                      <social.icon className="w-5 h-5 text-neutral-300 group-hover:text-white transition-colors duration-200" />
                    </motion.a>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Bottom Bar */}
          <motion.div variants={itemVariants} className="py-8">
            <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
              {/* Copyright */}
              <div className="flex items-center space-x-4 text-neutral-400 text-sm">
                <span>© 2024 OpenDentalScan. All rights reserved.</span>
                <div className="flex items-center space-x-1">
                  <span>Made with</span>
                  <HeartIcon className="w-4 h-4 text-red-400" />
                  <span>for the dental community</span>
                </div>
              </div>

              {/* Legal Links */}
              <div className="flex items-center space-x-6 text-neutral-400 text-sm">
                <motion.a
                  href="#privacy"
                  className="hover:text-white transition-colors duration-200"
                  whileHover={shouldReduceMotion ? {} : { y: -1 }}
                >
                  Privacy Policy
                </motion.a>
                <motion.a
                  href="#terms"
                  className="hover:text-white transition-colors duration-200"
                  whileHover={shouldReduceMotion ? {} : { y: -1 }}
                >
                  Terms of Service
                </motion.a>
                <motion.a
                  href="#license"
                  className="flex items-center space-x-1 hover:text-white transition-colors duration-200"
                  whileHover={shouldReduceMotion ? {} : { y: -1 }}
                >
                  <ShieldCheckIcon className="w-4 h-4" />
                  <span>MIT License</span>
                </motion.a>
              </div>
            </div>
          </motion.div>

          {/* Open Source Badge */}
          <motion.div 
            variants={itemVariants}
            className="text-center pb-8"
          >
            <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-dental-600/20 to-mint-600/20 border border-dental-400/30 rounded-full px-6 py-3">
              <BeakerIcon className="w-5 h-5 text-dental-300" />
              <span className="text-dental-200 font-medium">Proudly Open Source</span>
              <CodeBracketIcon className="w-5 h-5 text-mint-300" />
            </div>
          </motion.div>
        </motion.div>
      </div>
    </footer>
  )
}

export default Footer