'use client'

import { useState, useEffect } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { 
  Bars3Icon, 
  XMarkIcon,
  ArrowDownTrayIcon,
  DocumentTextIcon,
  BeakerIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'

const Navigation = () => {
  const [isOpen, setIsOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const shouldReduceMotion = useReducedMotion()

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const navItems = [
    { name: 'Features', href: '#features' },
    { name: 'Demo', href: '/demo' },
    { name: 'Pricing', href: '/pricing' },
    { name: 'Documentation', href: '/docs', icon: DocumentTextIcon },
    { name: 'Research', href: '/research', icon: BeakerIcon },
  ]

  const animationProps = shouldReduceMotion 
    ? {}
    : {
        initial: { y: -100 },
        animate: { y: 0 },
        transition: { duration: 0.4, ease: 'easeOut' }
      }

  return (
    <motion.nav 
      {...animationProps}
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrolled 
          ? 'bg-white/95 backdrop-blur-md shadow-soft border-b border-neutral-200' 
          : 'bg-transparent'
      }`}
    >
      <div className="container-max">
        <div className="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
          {/* Logo */}
          <motion.div 
            className="flex items-center space-x-3"
            whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <div className="w-10 h-10 bg-gradient-to-br from-dental-500 to-mint-500 rounded-xl flex items-center justify-center shadow-dental">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-dental-700 to-mint-600 bg-clip-text text-transparent">
                OpenDentalScan
              </h1>
              <p className="text-xs text-neutral-500 -mt-1">Professional</p>
            </div>
          </motion.div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item, index) => (
              <motion.a
                key={item.name}
                href={item.href}
                className="flex items-center space-x-1 text-neutral-600 hover:text-dental-600 font-medium transition-colors duration-200"
                whileHover={shouldReduceMotion ? {} : { y: -2 }}
                transition={{ duration: 0.2 }}
              >
                {item.icon && <item.icon className="w-4 h-4" />}
                <span>{item.name}</span>
              </motion.a>
            ))}
          </div>

          {/* CTA Buttons */}
          <div className="hidden md:flex items-center space-x-4">
            <Link
              href="/download"
              className="flex items-center space-x-2 bg-dental-600 hover:bg-dental-700 text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200 shadow-dental"
            >
              <ArrowDownTrayIcon className="w-4 h-4" />
              <span>Download</span>
            </Link>
            
            <Link
              href="/contact"
              className="text-neutral-600 hover:text-dental-600 font-medium transition-colors duration-200"
            >
              Contact
            </Link>
          </div>

          {/* Mobile menu button */}
          <motion.button
            className="md:hidden p-2 rounded-lg text-neutral-600 hover:text-dental-600 hover:bg-dental-50 transition-colors duration-200"
            onClick={() => setIsOpen(!isOpen)}
            whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
          >
            {isOpen ? (
              <XMarkIcon className="w-6 h-6" />
            ) : (
              <Bars3Icon className="w-6 h-6" />
            )}
          </motion.button>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <motion.div 
            className="md:hidden bg-white border-t border-neutral-200 shadow-gentle"
            initial={shouldReduceMotion ? {} : { opacity: 0, height: 0 }}
            animate={shouldReduceMotion ? {} : { opacity: 1, height: 'auto' }}
            exit={shouldReduceMotion ? {} : { opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="px-4 py-4 space-y-3">
              {navItems.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  className="flex items-center space-x-2 text-neutral-600 hover:text-dental-600 font-medium py-2 transition-colors duration-200"
                  onClick={() => setIsOpen(false)}
                >
                  {item.icon && <item.icon className="w-4 h-4" />}
                  <span>{item.name}</span>
                </a>
              ))}
              
              <div className="pt-4 border-t border-neutral-200 space-y-3">
                <Link
                  href="/download"
                  className="flex items-center justify-center space-x-2 bg-dental-600 text-white px-4 py-3 rounded-lg font-medium w-full"
                  onClick={() => setIsOpen(false)}
                >
                  <ArrowDownTrayIcon className="w-4 h-4" />
                  <span>Download</span>
                </Link>
                
                <Link
                  href="/contact"
                  className="block text-center text-neutral-600 font-medium py-2"
                  onClick={() => setIsOpen(false)}
                >
                  Contact
                </Link>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </motion.nav>
  )
}

export default Navigation