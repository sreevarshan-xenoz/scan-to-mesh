'use client'

import { useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import Link from 'next/link'
import { 
  ArrowLeftIcon,
  PhoneIcon,
  EnvelopeIcon,
  MapPinIcon,
  ClockIcon,
  ChatBubbleLeftRightIcon,
  DocumentTextIcon,
  AcademicCapIcon,
  CogIcon,
  CheckCircleIcon,
  ExclamationCircleIcon
} from '@heroicons/react/24/outline'

const ContactPage = () => {
  const shouldReduceMotion = useReducedMotion()
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    phone: '',
    subject: 'general',
    message: '',
    priority: 'medium'
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'success' | 'error'>('idle')

  const contactMethods = [
    {
      icon: PhoneIcon,
      title: 'Phone Support',
      description: 'Speak directly with our technical team',
      contact: '+1 (555) 123-4567',
      availability: 'Mon-Fri, 9AM-6PM EST',
      color: 'dental'
    },
    {
      icon: EnvelopeIcon,
      title: 'Email Support',
      description: 'Get detailed technical assistance',
      contact: 'support@opendentalscan.com',
      availability: '24/7 response within 4 hours',
      color: 'mint'
    },
    {
      icon: ChatBubbleLeftRightIcon,
      title: 'Live Chat',
      description: 'Instant help for quick questions',
      contact: 'Available on website',
      availability: 'Mon-Fri, 8AM-8PM EST',
      color: 'dental'
    },
    {
      icon: DocumentTextIcon,
      title: 'Support Portal',
      description: 'Access knowledge base and tickets',
      contact: 'portal.opendentalscan.com',
      availability: 'Available 24/7',
      color: 'mint'
    }
  ]

  const supportTypes = [
    {
      icon: CogIcon,
      title: 'Technical Support',
      description: 'Hardware setup, software installation, troubleshooting',
      responseTime: '< 2 hours'
    },
    {
      icon: AcademicCapIcon,
      title: 'Training & Education',
      description: 'Product training, best practices, workflow optimization',
      responseTime: '< 4 hours'
    },
    {
      icon: DocumentTextIcon,
      title: 'Sales & Licensing',
      description: 'Pricing, licensing, enterprise solutions, partnerships',
      responseTime: '< 1 hour'
    }
  ]

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    
    // Simulate form submission
    try {
      await new Promise(resolve => setTimeout(resolve, 2000))
      setSubmitStatus('success')
      setFormData({
        name: '',
        email: '',
        company: '',
        phone: '',
        subject: 'general',
        message: '',
        priority: 'medium'
      })
    } catch (error) {
      setSubmitStatus('error')
    } finally {
      setIsSubmitting(false)
    }
  }

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
                <ChatBubbleLeftRightIcon className="w-4 h-4" />
                <span>Contact & Support</span>
              </div>
              
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-neutral-900 mb-6">
                Get{' '}
                <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                  Expert Support
                </span>
              </h1>
              
              <p className="text-xl text-neutral-600 max-w-3xl mx-auto mb-8">
                Our technical team is here to help you succeed with OpenDentalScan. 
                Get professional support, training, and guidance for your dental scanning needs.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Contact Methods */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-16">
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">Multiple Ways to Reach Us</h2>
              <p className="text-neutral-600 max-w-2xl mx-auto">
                Choose the contact method that works best for you. Our team is ready to help.
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
              {contactMethods.map((method, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="bg-gradient-to-br from-white to-neutral-50 rounded-2xl p-6 border border-neutral-200 hover:border-dental-200 transition-all duration-300 hover:shadow-gentle text-center"
                >
                  <div className={`w-14 h-14 rounded-xl mb-4 flex items-center justify-center mx-auto ${
                    method.color === 'dental'
                      ? 'bg-gradient-to-br from-dental-500 to-dental-600'
                      : 'bg-gradient-to-br from-mint-500 to-mint-600'
                  } shadow-soft`}>
                    <method.icon className="w-7 h-7 text-white" />
                  </div>

                  <h3 className="text-lg font-bold text-neutral-900 mb-2">{method.title}</h3>
                  <p className="text-neutral-600 text-sm mb-4">{method.description}</p>
                  
                  <div className="space-y-2">
                    <div className={`font-semibold ${
                      method.color === 'dental' ? 'text-dental-700' : 'text-mint-700'
                    }`}>
                      {method.contact}
                    </div>
                    <div className="text-neutral-500 text-xs">{method.availability}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Contact Form & Support Types */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <div className="grid lg:grid-cols-2 gap-12">
              {/* Contact Form */}
              <motion.div variants={itemVariants}>
                <div className="bg-white rounded-2xl p-8 border border-neutral-200 shadow-soft">
                  <h2 className="text-2xl font-bold text-neutral-900 mb-6">Send Us a Message</h2>
                  
                  {submitStatus === 'success' && (
                    <div className="mb-6 p-4 bg-mint-50 border border-mint-200 rounded-lg flex items-center space-x-3">
                      <CheckCircleIcon className="w-5 h-5 text-mint-600" />
                      <span className="text-mint-700">Message sent successfully! We'll get back to you soon.</span>
                    </div>
                  )}

                  {submitStatus === 'error' && (
                    <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-3">
                      <ExclamationCircleIcon className="w-5 h-5 text-red-600" />
                      <span className="text-red-700">Failed to send message. Please try again or contact us directly.</span>
                    </div>
                  )}

                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                          Full Name *
                        </label>
                        <input
                          type="text"
                          name="name"
                          value={formData.name}
                          onChange={handleInputChange}
                          required
                          className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-colors duration-200"
                          placeholder="Your full name"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                          Email Address *
                        </label>
                        <input
                          type="email"
                          name="email"
                          value={formData.email}
                          onChange={handleInputChange}
                          required
                          className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-colors duration-200"
                          placeholder="your.email@example.com"
                        />
                      </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                          Company/Organization
                        </label>
                        <input
                          type="text"
                          name="company"
                          value={formData.company}
                          onChange={handleInputChange}
                          className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-colors duration-200"
                          placeholder="Your company name"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                          Phone Number
                        </label>
                        <input
                          type="tel"
                          name="phone"
                          value={formData.phone}
                          onChange={handleInputChange}
                          className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-colors duration-200"
                          placeholder="+1 (555) 123-4567"
                        />
                      </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                          Subject *
                        </label>
                        <select
                          name="subject"
                          value={formData.subject}
                          onChange={handleInputChange}
                          required
                          className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-colors duration-200"
                        >
                          <option value="general">General Inquiry</option>
                          <option value="technical">Technical Support</option>
                          <option value="sales">Sales & Pricing</option>
                          <option value="training">Training & Education</option>
                          <option value="partnership">Partnership</option>
                          <option value="feedback">Feedback</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 mb-2">
                          Priority
                        </label>
                        <select
                          name="priority"
                          value={formData.priority}
                          onChange={handleInputChange}
                          className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-colors duration-200"
                        >
                          <option value="low">Low</option>
                          <option value="medium">Medium</option>
                          <option value="high">High</option>
                          <option value="urgent">Urgent</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-neutral-700 mb-2">
                        Message *
                      </label>
                      <textarea
                        name="message"
                        value={formData.message}
                        onChange={handleInputChange}
                        required
                        rows={6}
                        className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-colors duration-200 resize-vertical"
                        placeholder="Please describe your question or issue in detail..."
                      />
                    </div>

                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="w-full bg-dental-600 hover:bg-dental-700 disabled:bg-dental-400 text-white py-4 px-6 rounded-xl font-semibold text-lg transition-colors duration-200 shadow-dental"
                    >
                      {isSubmitting ? 'Sending Message...' : 'Send Message'}
                    </button>
                  </form>
                </div>
              </motion.div>

              {/* Support Types */}
              <motion.div variants={itemVariants} className="space-y-8">
                <div>
                  <h2 className="text-2xl font-bold text-neutral-900 mb-6">Support Categories</h2>
                  <p className="text-neutral-600 mb-8">
                    Our specialized support teams are ready to help with different aspects of OpenDentalScan.
                  </p>
                </div>

                <div className="space-y-6">
                  {supportTypes.map((type, index) => (
                    <div
                      key={index}
                      className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-soft"
                    >
                      <div className="flex items-start space-x-4">
                        <div className="w-12 h-12 bg-gradient-to-br from-dental-500 to-mint-500 rounded-xl flex items-center justify-center flex-shrink-0">
                          <type.icon className="w-6 h-6 text-white" />
                        </div>
                        
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-2">
                            <h3 className="text-lg font-bold text-neutral-900">{type.title}</h3>
                            <span className="text-sm text-mint-600 font-medium">{type.responseTime}</span>
                          </div>
                          <p className="text-neutral-600">{type.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Emergency Contact */}
                <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-2xl p-6 border border-red-200">
                  <h3 className="text-lg font-bold text-neutral-900 mb-2 flex items-center space-x-2">
                    <ExclamationCircleIcon className="w-5 h-5 text-red-600" />
                    <span>Emergency Support</span>
                  </h3>
                  <p className="text-neutral-700 mb-4">
                    For critical issues affecting patient care or urgent technical problems:
                  </p>
                  <div className="space-y-2">
                    <div className="font-semibold text-red-700">Emergency Hotline: +1 (555) 911-SCAN</div>
                    <div className="text-red-600 text-sm">Available 24/7 for critical issues</div>
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Office Information */}
      <section className="py-16 bg-gradient-to-br from-neutral-900 to-dental-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4">Our Locations</h2>
              <p className="text-neutral-300 max-w-2xl mx-auto">
                Global presence with local support for dental professionals worldwide
              </p>
            </motion.div>

            <div className="grid md:grid-cols-3 gap-8">
              <motion.div variants={itemVariants} className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-dental-400 to-dental-500 rounded-2xl mx-auto mb-6 flex items-center justify-center">
                  <MapPinIcon className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-semibold mb-4">North America HQ</h3>
                <div className="space-y-2 text-neutral-300">
                  <p>123 Innovation Drive</p>
                  <p>San Francisco, CA 94105</p>
                  <p>United States</p>
                </div>
              </motion.div>

              <motion.div variants={itemVariants} className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-mint-400 to-mint-500 rounded-2xl mx-auto mb-6 flex items-center justify-center">
                  <ClockIcon className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Support Hours</h3>
                <div className="space-y-2 text-neutral-300">
                  <p>Monday - Friday: 8AM - 8PM EST</p>
                  <p>Saturday: 10AM - 4PM EST</p>
                  <p>Sunday: Emergency only</p>
                </div>
              </motion.div>

              <motion.div variants={itemVariants} className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-dental-400 to-mint-400 rounded-2xl mx-auto mb-6 flex items-center justify-center">
                  <ChatBubbleLeftRightIcon className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Response Times</h3>
                <div className="space-y-2 text-neutral-300">
                  <p>Email: Within 4 hours</p>
                  <p>Phone: Immediate</p>
                  <p>Emergency: Within 15 minutes</p>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default ContactPage