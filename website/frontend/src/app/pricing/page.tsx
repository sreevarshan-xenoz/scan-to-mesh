'use client'

import { useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import Link from 'next/link'
import { 
  ArrowLeftIcon,
  CheckIcon,
  XMarkIcon,
  StarIcon,
  CurrencyDollarIcon,
  BuildingOfficeIcon,
  AcademicCapIcon,
  BeakerIcon,
  ShieldCheckIcon,
  PhoneIcon,
  ChatBubbleLeftRightIcon,
  ClockIcon,
  CpuChipIcon,
  CameraIcon,
  CloudIcon,
  UsersIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline'

const PricingPage = () => {
  const shouldReduceMotion = useReducedMotion()
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'annual'>('annual')

  const pricingTiers = [
    {
      name: 'Starter',
      description: 'Perfect for small dental practices getting started',
      icon: AcademicCapIcon,
      color: 'neutral',
      popular: false,
      pricing: {
        monthly: 299,
        annual: 2390, // ~20% discount
        setup: 0
      },
      features: {
        included: [
          'Single workstation license',
          'Basic 3D reconstruction',
          'Standard AI analysis',
          'STL/OBJ export',
          'Email support',
          'Basic training (2 hours)',
          'Intel RealSense D435i support',
          'Standard resolution scanning',
          'Basic reporting',
          'Software updates'
        ],
        excluded: [
          'Advanced AI models',
          'Multi-device setup',
          'Priority support',
          'Custom training',
          'Enterprise features'
        ]
      },
      limits: {
        'Workstations': '1',
        'Monthly Scans': '500',
        'Storage': '50 GB',
        'Support': 'Email only'
      }
    },
    {
      name: 'Professional',
      description: 'Advanced features for growing dental practices',
      icon: BuildingOfficeIcon,
      color: 'dental',
      popular: true,
      pricing: {
        monthly: 599,
        annual: 4790, // ~33% discount
        setup: 199
      },
      features: {
        included: [
          'Up to 5 workstation licenses',
          'Advanced 3D reconstruction',
          'Full AI analysis suite',
          'All export formats (STL, OBJ, PLY, DICOM)',
          'Priority phone & email support',
          'Comprehensive training (8 hours)',
          'Multi-device support (RealSense, stereo cameras)',
          'High-resolution scanning',
          'Advanced reporting & analytics',
          'Custom workflow integration',
          'Performance optimization',
          'Quarterly check-ins'
        ],
        excluded: [
          'Unlimited workstations',
          'Custom AI training',
          'On-site support',
          'White-label options'
        ]
      },
      limits: {
        'Workstations': '5',
        'Monthly Scans': '2,500',
        'Storage': '500 GB',
        'Support': 'Phone + Email'
      }
    },
    {
      name: 'Enterprise',
      description: 'Complete solution for large organizations and chains',
      icon: BeakerIcon,
      color: 'mint',
      popular: false,
      pricing: {
        monthly: 1299,
        annual: 10390, // ~33% discount
        setup: 999
      },
      features: {
        included: [
          'Unlimited workstation licenses',
          'Enterprise 3D reconstruction',
          'Custom AI model training',
          'All export formats + custom formats',
          'Dedicated support manager',
          'On-site training & setup',
          'All hardware support + custom integration',
          'Ultra-high resolution scanning',
          'Enterprise reporting & dashboards',
          'Custom workflow development',
          'Performance monitoring & optimization',
          'Monthly strategic reviews',
          'White-label options',
          'API access & custom integrations',
          'Multi-location management',
          'Advanced security & compliance'
        ],
        excluded: []
      },
      limits: {
        'Workstations': 'Unlimited',
        'Monthly Scans': 'Unlimited',
        'Storage': 'Unlimited',
        'Support': 'Dedicated Manager'
      }
    }
  ]

  const addOns = [
    {
      name: 'Additional Workstation',
      description: 'Add more scanning stations to your setup',
      price: { monthly: 99, annual: 790 },
      icon: CpuChipIcon
    },
    {
      name: 'Premium Hardware Bundle',
      description: 'Intel RealSense L515 + calibration kit',
      price: { monthly: 0, annual: 1499 },
      icon: CameraIcon
    },
    {
      name: 'Cloud Storage (1TB)',
      description: 'Secure cloud backup and sync',
      price: { monthly: 49, annual: 390 },
      icon: CloudIcon
    },
    {
      name: 'Advanced Training',
      description: 'On-site training and workflow optimization',
      price: { monthly: 0, annual: 2999 },
      icon: AcademicCapIcon
    }
  ]

  const comparisonFeatures = [
    { category: 'Core Features', features: [
      'Real-time 3D reconstruction',
      'AI-powered analysis',
      'Multi-format export',
      'Hardware compatibility',
      'Software updates'
    ]},
    { category: 'Support & Training', features: [
      'Email support',
      'Phone support',
      'Live chat support',
      'On-site training',
      'Dedicated support manager'
    ]},
    { category: 'Advanced Features', features: [
      'Custom AI training',
      'API access',
      'White-label options',
      'Multi-location management',
      'Enterprise security'
    ]}
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

  const getPrice = (tier: typeof pricingTiers[0]) => {
    return billingCycle === 'monthly' ? tier.pricing.monthly : Math.round(tier.pricing.annual / 12)
  }

  const getTotalAnnual = (tier: typeof pricingTiers[0]) => {
    return billingCycle === 'monthly' ? tier.pricing.monthly * 12 : tier.pricing.annual
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
                <CurrencyDollarIcon className="w-4 h-4" />
                <span>Pricing & Plans</span>
              </div>
              
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-neutral-900 mb-6">
                Choose Your{' '}
                <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                  Perfect Plan
                </span>
              </h1>
              
              <p className="text-xl text-neutral-600 max-w-3xl mx-auto mb-8">
                Professional dental scanning solutions for every practice size. 
                Start with our flexible plans and scale as you grow.
              </p>

              {/* Billing Toggle */}
              <div className="inline-flex items-center bg-white rounded-xl p-2 border border-neutral-200 shadow-soft">
                <button
                  onClick={() => setBillingCycle('monthly')}
                  className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 ${
                    billingCycle === 'monthly'
                      ? 'bg-dental-600 text-white shadow-dental'
                      : 'text-neutral-600 hover:text-dental-600'
                  }`}
                >
                  Monthly
                </button>
                <button
                  onClick={() => setBillingCycle('annual')}
                  className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 relative ${
                    billingCycle === 'annual'
                      ? 'bg-dental-600 text-white shadow-dental'
                      : 'text-neutral-600 hover:text-dental-600'
                  }`}
                >
                  Annual
                  <span className="absolute -top-2 -right-2 bg-mint-500 text-white text-xs px-2 py-1 rounded-full">
                    Save 33%
                  </span>
                </button>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Pricing Tiers */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <div className="grid lg:grid-cols-3 gap-8">
              {pricingTiers.map((tier, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className={`relative bg-white rounded-3xl p-8 border-2 transition-all duration-300 hover:shadow-gentle ${
                    tier.popular
                      ? 'border-dental-300 shadow-dental scale-105'
                      : 'border-neutral-200 hover:border-dental-200'
                  }`}
                >
                  {/* Popular Badge */}
                  {tier.popular && (
                    <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                      <div className="bg-gradient-to-r from-dental-500 to-mint-500 text-white px-6 py-2 rounded-full text-sm font-semibold shadow-dental flex items-center space-x-2">
                        <StarIcon className="w-4 h-4" />
                        <span>Most Popular</span>
                      </div>
                    </div>
                  )}

                  {/* Header */}
                  <div className="text-center mb-8">
                    <div className={`w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center ${
                      tier.color === 'dental' ? 'bg-gradient-to-br from-dental-500 to-dental-600' :
                      tier.color === 'mint' ? 'bg-gradient-to-br from-mint-500 to-mint-600' :
                      'bg-gradient-to-br from-neutral-500 to-neutral-600'
                    } shadow-soft`}>
                      <tier.icon className="w-8 h-8 text-white" />
                    </div>
                    
                    <h3 className="text-2xl font-bold text-neutral-900 mb-2">{tier.name}</h3>
                    <p className="text-neutral-600 mb-6">{tier.description}</p>
                    
                    {/* Pricing */}
                    <div className="mb-6">
                      <div className="flex items-baseline justify-center space-x-2">
                        <span className="text-4xl font-bold text-neutral-900">
                          ${getPrice(tier).toLocaleString()}
                        </span>
                        <span className="text-neutral-500">/month</span>
                      </div>
                      
                      {billingCycle === 'annual' && (
                        <div className="text-sm text-neutral-500 mt-2">
                          ${getTotalAnnual(tier).toLocaleString()} billed annually
                        </div>
                      )}
                      
                      {tier.pricing.setup > 0 && (
                        <div className="text-sm text-neutral-500 mt-1">
                          + ${tier.pricing.setup} setup fee
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Features */}
                  <div className="space-y-4 mb-8">
                    {/* Limits */}
                    <div className="bg-neutral-50 rounded-lg p-4">
                      <h4 className="font-semibold text-neutral-900 mb-3">Plan Limits</h4>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        {Object.entries(tier.limits).map(([key, value]) => (
                          <div key={key}>
                            <div className="text-neutral-500">{key}</div>
                            <div className="font-medium text-neutral-900">{value}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Included Features */}
                    <div>
                      <h4 className="font-semibold text-neutral-900 mb-3">Included Features</h4>
                      <div className="space-y-2">
                        {tier.features.included.slice(0, 6).map((feature, featureIndex) => (
                          <div key={featureIndex} className="flex items-center space-x-3">
                            <CheckIcon className="w-4 h-4 text-mint-500 flex-shrink-0" />
                            <span className="text-neutral-700 text-sm">{feature}</span>
                          </div>
                        ))}
                        {tier.features.included.length > 6 && (
                          <div className="text-sm text-neutral-500 mt-2">
                            + {tier.features.included.length - 6} more features
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* CTA Button */}
                  <button
                    className={`w-full py-4 px-6 rounded-xl font-semibold text-lg transition-all duration-200 ${
                      tier.popular
                        ? 'bg-dental-600 hover:bg-dental-700 text-white shadow-dental'
                        : 'bg-neutral-100 hover:bg-neutral-200 text-neutral-900'
                    }`}
                  >
                    {tier.name === 'Enterprise' ? 'Contact Sales' : 'Start Free Trial'}
                  </button>
                  
                  <div className="text-center mt-4">
                    <Link
                      href="/contact"
                      className="text-dental-600 hover:text-dental-700 text-sm font-medium"
                    >
                      Questions? Contact us →
                    </Link>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Add-ons */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-16">
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">Add-Ons & Extras</h2>
              <p className="text-neutral-600 max-w-2xl mx-auto">
                Enhance your OpenDentalScan experience with additional features and services
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {addOns.map((addon, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="bg-gradient-to-br from-white to-neutral-50 rounded-2xl p-6 border border-neutral-200 hover:border-dental-200 transition-all duration-300 hover:shadow-gentle"
                >
                  <div className="w-12 h-12 bg-gradient-to-br from-dental-500 to-mint-500 rounded-xl mb-4 flex items-center justify-center">
                    <addon.icon className="w-6 h-6 text-white" />
                  </div>
                  
                  <h3 className="text-lg font-bold text-neutral-900 mb-2">{addon.name}</h3>
                  <p className="text-neutral-600 text-sm mb-4">{addon.description}</p>
                  
                  <div className="text-2xl font-bold text-dental-600">
                    {addon.price.monthly > 0 && billingCycle === 'monthly' 
                      ? `$${addon.price.monthly}/mo`
                      : `$${addon.price.annual}`
                    }
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* FAQ */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-16">
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">Frequently Asked Questions</h2>
              <p className="text-neutral-600">
                Common questions about our pricing and plans
              </p>
            </motion.div>

            <motion.div variants={itemVariants} className="space-y-6">
              {[
                {
                  q: "Is there a free trial available?",
                  a: "Yes! We offer a 30-day free trial for all plans. No credit card required to start."
                },
                {
                  q: "Can I change plans later?",
                  a: "Absolutely. You can upgrade or downgrade your plan at any time. Changes take effect at your next billing cycle."
                },
                {
                  q: "What's included in the setup fee?",
                  a: "Setup includes software installation, hardware configuration, initial training, and workflow optimization."
                },
                {
                  q: "Do you offer educational discounts?",
                  a: "Yes, we provide special pricing for dental schools and educational institutions. Contact us for details."
                },
                {
                  q: "What happens if I exceed my scan limits?",
                  a: "We'll notify you when approaching limits. You can upgrade your plan or purchase additional scan credits."
                },
                {
                  q: "Is support included in all plans?",
                  a: "Yes, all plans include support. Higher tiers get priority support and dedicated account management."
                }
              ].map((faq, index) => (
                <div key={index} className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-soft">
                  <h3 className="text-lg font-bold text-neutral-900 mb-3">{faq.q}</h3>
                  <p className="text-neutral-600">{faq.a}</p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 bg-gradient-to-br from-dental-600 to-mint-500 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants}>
              <h2 className="text-3xl font-bold mb-4">
                Ready to Transform Your Dental Practice?
              </h2>
              <p className="text-xl mb-8 opacity-90">
                Join hundreds of dental professionals already using OpenDentalScan
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button className="bg-white text-dental-600 px-8 py-4 rounded-xl font-semibold text-lg hover:bg-neutral-50 transition-colors duration-200 shadow-lg">
                  Start Free Trial
                </button>
                <Link
                  href="/contact"
                  className="bg-transparent border-2 border-white text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-white/10 transition-colors duration-200 flex items-center justify-center space-x-2"
                >
                  <span>Talk to Sales</span>
                  <ArrowRightIcon className="w-5 h-5" />
                </Link>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default PricingPage