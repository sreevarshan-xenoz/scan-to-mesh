'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { 
  CubeIcon,
  CpuChipIcon,
  EyeIcon,
  ShieldCheckIcon,
  CloudArrowUpIcon,
  DocumentArrowDownIcon,
  BeakerIcon,
  CameraIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  GlobeAltIcon,
  AcademicCapIcon
} from '@heroicons/react/24/outline'

const Features = () => {
  const shouldReduceMotion = useReducedMotion()

  const features = [
    {
      icon: CubeIcon,
      title: 'Real-Time 3D Reconstruction',
      description: 'GPU-accelerated TSDF fusion delivers professional-grade 3D models with sub-millimeter accuracy at 30 FPS.',
      color: 'dental',
      stats: '30 FPS • 0.1mm precision'
    },
    {
      icon: CpuChipIcon,
      title: 'AI-Powered Analysis',
      description: 'Advanced neural networks for tooth segmentation, pathology detection, and automated dental numbering.',
      color: 'mint',
      stats: '95%+ accuracy • 22 AI models'
    },
    {
      icon: CameraIcon,
      title: 'Multi-Hardware Support',
      description: 'Works with Intel RealSense, stereo cameras, webcams, and custom structured light systems.',
      color: 'dental',
      stats: '15+ camera types supported'
    },
    {
      icon: ShieldCheckIcon,
      title: 'Clinical Grade Security',
      description: 'HIPAA-compliant data handling, encrypted storage, and complete patient privacy protection.',
      color: 'mint',
      stats: 'HIPAA • GDPR compliant'
    },
    {
      icon: DocumentArrowDownIcon,
      title: 'Professional Export',
      description: 'Export to STL, OBJ, PLY, DICOM formats for seamless CAD/CAM and clinical workflow integration.',
      color: 'dental',
      stats: '6+ export formats'
    },
    {
      icon: GlobeAltIcon,
      title: 'Modern Technology Stack',
      description: 'Built with cutting-edge open-source technologies for reliability, performance, and future-proof innovation.',
      color: 'mint',
      stats: 'Latest tech stack • Future-ready'
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
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: 'easeOut' }
    }
  }

  return (
    <section id="features" className="section-padding bg-white">
      <div className="container-max">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          {/* Section Header */}
          <motion.div variants={itemVariants} className="text-center mb-16">
            <div className="inline-flex items-center space-x-2 bg-dental-100 text-dental-700 px-4 py-2 rounded-full text-sm font-medium mb-4">
              <BeakerIcon className="w-4 h-4" />
              <span>Advanced Features</span>
            </div>
            
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-neutral-900 mb-6">
              Everything You Need for{' '}
              <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                Professional Scanning
              </span>
            </h2>
            
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
              Built using cutting-edge open-source technologies, 
              delivering enterprise-grade capabilities with modern innovation and cost-effectiveness.
            </p>
          </motion.div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                className="group relative"
              >
                <div className="h-full bg-gradient-to-br from-white to-neutral-50 rounded-2xl p-8 border border-neutral-200 hover:border-dental-200 transition-all duration-300 hover:shadow-gentle">
                  {/* Icon */}
                  <div className={`inline-flex items-center justify-center w-14 h-14 rounded-xl mb-6 ${
                    feature.color === 'dental' 
                      ? 'bg-gradient-to-br from-dental-500 to-dental-600' 
                      : 'bg-gradient-to-br from-mint-500 to-mint-600'
                  } shadow-soft`}>
                    <feature.icon className="w-7 h-7 text-white" />
                  </div>

                  {/* Content */}
                  <div className="space-y-4">
                    <h3 className="text-xl font-bold text-neutral-900 group-hover:text-dental-700 transition-colors duration-200">
                      {feature.title}
                    </h3>
                    
                    <p className="text-neutral-600 leading-relaxed">
                      {feature.description}
                    </p>

                    {/* Stats Badge */}
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                      feature.color === 'dental'
                        ? 'bg-dental-50 text-dental-700'
                        : 'bg-mint-50 text-mint-700'
                    }`}>
                      {feature.stats}
                    </div>
                  </div>

                  {/* Hover Effect */}
                  <div className={`absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 ${
                    feature.color === 'dental'
                      ? 'bg-gradient-to-br from-dental-50/50 to-transparent'
                      : 'bg-gradient-to-br from-mint-50/50 to-transparent'
                  }`} />
                </div>
              </motion.div>
            ))}
          </div>

          {/* Bottom CTA */}
          <motion.div variants={itemVariants} className="text-center mt-16">
            <div className="bg-gradient-to-r from-dental-50 to-mint-50 rounded-2xl p-8 border border-dental-100">
              <div className="flex flex-col md:flex-row items-center justify-between space-y-6 md:space-y-0">
                <div className="text-left">
                  <h3 className="text-2xl font-bold text-neutral-900 mb-2">
                    Ready to Experience Professional Dental Scanning?
                  </h3>
                  <p className="text-neutral-600">
                    Download our open-source solution and start scanning in minutes.
                  </p>
                </div>
                
                <div className="flex flex-col sm:flex-row gap-4">
                  <motion.button
                    className="flex items-center space-x-2 bg-dental-600 hover:bg-dental-700 text-white px-6 py-3 rounded-xl font-semibold transition-colors duration-200 shadow-dental"
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    <DocumentArrowDownIcon className="w-5 h-5" />
                    <span>Download Now</span>
                  </motion.button>
                  
                  <motion.button
                    className="flex items-center space-x-2 bg-white hover:bg-neutral-50 text-dental-600 px-6 py-3 rounded-xl font-semibold border border-dental-200 hover:border-dental-300 transition-all duration-200"
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    <AcademicCapIcon className="w-5 h-5" />
                    <span>View Documentation</span>
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

export default Features