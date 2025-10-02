'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { 
  CheckCircleIcon,
  XCircleIcon,
  CurrencyDollarIcon,
  ShieldCheckIcon,
  CogIcon,
  GlobeAltIcon,
  BeakerIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline'

const Comparison = () => {
  const shouldReduceMotion = useReducedMotion()

  const comparisonData = [
    {
      category: 'Cost & Licensing',
      icon: CurrencyDollarIcon,
      items: [
        {
          feature: 'Hardware Cost',
          openSource: '$200-500',
          commercial: '$50,000+',
          advantage: 'open'
        },
        {
          feature: 'Software License',
          openSource: 'Free (MIT)',
          commercial: '$10,000+/year',
          advantage: 'open'
        },
        {
          feature: 'Updates & Support',
          openSource: 'Community Driven',
          commercial: 'Vendor Controlled',
          advantage: 'open'
        },
        {
          feature: 'Per-Seat Licensing',
          openSource: 'Unlimited',
          commercial: 'Per User Fee',
          advantage: 'open'
        }
      ]
    },
    {
      category: 'Technical Capabilities',
      icon: CogIcon,
      items: [
        {
          feature: '3D Reconstruction',
          openSource: 'PyTorch GPU TSDF',
          commercial: 'Proprietary CUDA',
          advantage: 'equal'
        },
        {
          feature: 'AI/ML Models',
          openSource: 'Open ONNX Models',
          commercial: 'Encrypted Models',
          advantage: 'open'
        },
        {
          feature: 'Processing Speed',
          openSource: '30+ FPS',
          commercial: '30-60 FPS',
          advantage: 'commercial'
        },
        {
          feature: 'Accuracy',
          openSource: '0.1mm precision',
          commercial: '0.05-0.1mm',
          advantage: 'equal'
        }
      ]
    },
    {
      category: 'Flexibility & Control',
      icon: ShieldCheckIcon,
      items: [
        {
          feature: 'Technology Stack',
          openSource: 'Modern & Flexible',
          commercial: 'Legacy Systems',
          advantage: 'open'
        },
        {
          feature: 'Customization',
          openSource: 'Unlimited',
          commercial: 'Limited/None',
          advantage: 'open'
        },
        {
          feature: 'Data Ownership',
          openSource: 'Your Infrastructure',
          commercial: 'Vendor Servers',
          advantage: 'open'
        },
        {
          feature: 'Vendor Lock-in',
          openSource: 'None',
          commercial: 'Complete',
          advantage: 'open'
        }
      ]
    },
    {
      category: 'Hardware Support',
      icon: BeakerIcon,
      items: [
        {
          feature: 'Camera Options',
          openSource: '15+ Types',
          commercial: 'Proprietary Only',
          advantage: 'open'
        },
        {
          feature: 'Intel RealSense',
          openSource: 'Full Support',
          commercial: 'Not Supported',
          advantage: 'open'
        },
        {
          feature: 'Custom Hardware',
          openSource: 'Easy Integration',
          commercial: 'Not Possible',
          advantage: 'open'
        },
        {
          feature: 'Upgrade Path',
          openSource: 'Any Hardware',
          commercial: 'Vendor Hardware',
          advantage: 'open'
        }
      ]
    }
  ]

  const marketComparison = [
    {
      name: 'OpenDentalScan',
      type: 'Professional',
      price: '$500',
      color: 'dental',
      features: [
        'Modern technology stack',
        'Multiple hardware support',
        'Advanced AI capabilities',
        'Flexible configuration',
        'Professional support',
        'Cost-effective solution'
      ]
    },
    {
      name: 'IntraoralScan 3.5',
      type: 'Commercial',
      price: '$50,000+',
      color: 'neutral',
      features: [
        'Proprietary software',
        'Single hardware option',
        'Encrypted AI models',
        'Limited customization',
        'Vendor support only',
        'Complete vendor lock-in'
      ]
    },
    {
      name: 'iTero Element',
      type: 'Commercial',
      price: '$40,000+',
      color: 'neutral',
      features: [
        'Closed source',
        'Proprietary hardware',
        'No AI transparency',
        'No customization',
        'Expensive support',
        'Vendor dependency'
      ]
    }
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

  const getAdvantageIcon = (advantage: string) => {
    switch (advantage) {
      case 'open':
        return <CheckCircleIcon className="w-5 h-5 text-mint-500" />
      case 'commercial':
        return <CheckCircleIcon className="w-5 h-5 text-orange-500" />
      case 'equal':
        return <div className="w-5 h-5 rounded-full bg-neutral-300" />
      default:
        return null
    }
  }

  return (
    <section id="comparison" className="section-padding bg-gradient-to-br from-neutral-50 to-dental-50">
      <div className="container-max">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          {/* Section Header */}
          <motion.div variants={itemVariants} className="text-center mb-16">
            <div className="inline-flex items-center space-x-2 bg-white text-dental-700 px-4 py-2 rounded-full text-sm font-medium mb-4 shadow-soft">
              <ChartBarIcon className="w-4 h-4" />
              <span>Market Comparison</span>
            </div>
            
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-neutral-900 mb-6">
              Why Choose{' '}
              <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                OpenDentalScan?
              </span>
            </h2>
            
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
              Compare OpenDentalScan with commercial solutions. See how open-source 
              delivers professional capabilities at a fraction of the cost.
            </p>
          </motion.div>

          {/* Market Overview Cards */}
          <motion.div variants={itemVariants} className="grid md:grid-cols-3 gap-8 mb-16">
            {marketComparison.map((solution, index) => (
              <div
                key={index}
                className={`relative bg-white rounded-2xl p-8 border-2 transition-all duration-300 ${
                  solution.color === 'dental'
                    ? 'border-dental-200 shadow-dental hover:shadow-gentle'
                    : 'border-neutral-200 hover:border-neutral-300 shadow-soft hover:shadow-gentle'
                }`}
              >
                {/* Recommended Badge */}
                {solution.color === 'dental' && (
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                    <div className="bg-gradient-to-r from-dental-500 to-mint-500 text-white px-4 py-2 rounded-full text-sm font-semibold shadow-dental">
                      Recommended
                    </div>
                  </div>
                )}

                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-neutral-900 mb-2">
                    {solution.name}
                  </h3>
                  <div className={`text-sm font-medium px-3 py-1 rounded-full inline-block mb-4 ${
                    solution.color === 'dental'
                      ? 'bg-dental-100 text-dental-700'
                      : 'bg-neutral-100 text-neutral-700'
                  }`}>
                    {solution.type}
                  </div>
                  <div className="text-3xl font-bold text-neutral-900">
                    {solution.price}
                  </div>
                </div>

                <div className="space-y-3">
                  {solution.features.map((feature, featureIndex) => (
                    <div key={featureIndex} className="flex items-center space-x-3">
                      {solution.color === 'dental' ? (
                        <CheckCircleIcon className="w-5 h-5 text-mint-500 flex-shrink-0" />
                      ) : (
                        <XCircleIcon className="w-5 h-5 text-neutral-400 flex-shrink-0" />
                      )}
                      <span className={`text-sm ${
                        solution.color === 'dental' ? 'text-neutral-700' : 'text-neutral-500'
                      }`}>
                        {feature}
                      </span>
                    </div>
                  ))}
                </div>

                {solution.color === 'dental' && (
                  <motion.button
                    className="w-full mt-6 bg-dental-600 hover:bg-dental-700 text-white py-3 rounded-xl font-semibold transition-colors duration-200"
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    Get Started Free
                  </motion.button>
                )}
              </div>
            ))}
          </motion.div>

          {/* Detailed Comparison Table */}
          <motion.div variants={itemVariants} className="bg-white rounded-3xl shadow-gentle border border-neutral-200 overflow-hidden">
            <div className="p-8">
              <h3 className="text-2xl font-bold text-neutral-900 mb-8 text-center">
                Detailed Feature Comparison
              </h3>

              <div className="space-y-12">
                {comparisonData.map((category, categoryIndex) => (
                  <div key={categoryIndex}>
                    {/* Category Header */}
                    <div className="flex items-center space-x-3 mb-6">
                      <div className="w-10 h-10 bg-gradient-to-br from-dental-500 to-mint-500 rounded-xl flex items-center justify-center">
                        <category.icon className="w-5 h-5 text-white" />
                      </div>
                      <h4 className="text-xl font-bold text-neutral-900">
                        {category.category}
                      </h4>
                    </div>

                    {/* Comparison Items */}
                    <div className="grid gap-4">
                      {/* Header Row */}
                      <div className="grid grid-cols-4 gap-4 pb-4 border-b border-neutral-200">
                        <div className="font-semibold text-neutral-700">Feature</div>
                        <div className="font-semibold text-dental-700 text-center">OpenDentalScan</div>
                        <div className="font-semibold text-neutral-700 text-center">Commercial</div>
                        <div className="font-semibold text-neutral-700 text-center">Advantage</div>
                      </div>

                      {/* Data Rows */}
                      {category.items.map((item, itemIndex) => (
                        <div key={itemIndex} className="grid grid-cols-4 gap-4 py-3 hover:bg-neutral-50 rounded-lg transition-colors duration-200">
                          <div className="text-neutral-700 font-medium">
                            {item.feature}
                          </div>
                          <div className="text-center">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              item.advantage === 'open'
                                ? 'bg-mint-100 text-mint-700'
                                : item.advantage === 'equal'
                                ? 'bg-dental-100 text-dental-700'
                                : 'bg-neutral-100 text-neutral-700'
                            }`}>
                              {item.openSource}
                            </span>
                          </div>
                          <div className="text-center">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                              item.advantage === 'commercial'
                                ? 'bg-orange-100 text-orange-700'
                                : 'bg-neutral-100 text-neutral-700'
                            }`}>
                              {item.commercial}
                            </span>
                          </div>
                          <div className="flex justify-center">
                            {getAdvantageIcon(item.advantage)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Bottom Summary */}
          <motion.div variants={itemVariants} className="mt-16 text-center">
            <div className="bg-gradient-to-r from-dental-600 to-mint-500 rounded-3xl p-8 text-white">
              <h3 className="text-3xl font-bold mb-4">
                The Clear Choice for Modern Dental Practices
              </h3>
              <p className="text-xl mb-8 opacity-90 max-w-3xl mx-auto">
                OpenDentalScan delivers professional-grade capabilities with modern technology, 
                flexible configuration, and 100x cost savings compared to traditional solutions.
              </p>
              
              <div className="grid md:grid-cols-3 gap-8 mb-8">
                <div className="text-center">
                  <div className="text-4xl font-bold mb-2">100x</div>
                  <div className="text-lg opacity-90">Cost Reduction</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold mb-2">100%</div>
                  <div className="text-lg opacity-90">Professional</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold mb-2">0</div>
                  <div className="text-lg opacity-90">Vendor Lock-in</div>
                </div>
              </div>

              <motion.button
                className="bg-white text-dental-600 px-8 py-4 rounded-xl font-bold text-lg hover:bg-neutral-50 transition-colors duration-200 shadow-lg"
                whileHover={shouldReduceMotion ? {} : { scale: 1.05 }}
                whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
              >
                Start Your Free Trial Today
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

export default Comparison