'use client'

import { motion, useReducedMotion } from 'framer-motion'
import Link from 'next/link'
import { 
  AcademicCapIcon,
  BeakerIcon,
  ChartBarIcon,
  CpuChipIcon,
  CubeIcon,
  EyeIcon,
  DocumentTextIcon,
  ArrowTopRightOnSquareIcon,
  ArrowLeftIcon,
  CalendarIcon,
  UserGroupIcon,
  TrophyIcon,
  LightBulbIcon
} from '@heroicons/react/24/outline'

const ResearchPage = () => {
  const shouldReduceMotion = useReducedMotion()

  const researchAreas = [
    {
      title: '3D Reconstruction Algorithms',
      icon: CubeIcon,
      color: 'dental',
      description: 'Advanced volumetric fusion and surface reconstruction techniques',
      achievements: [
        'Sub-millimeter accuracy TSDF fusion',
        'Real-time mesh generation at 30+ FPS',
        'GPU-accelerated processing pipeline',
        'Adaptive quality control algorithms'
      ]
    },
    {
      title: 'AI-Powered Dental Analysis',
      icon: CpuChipIcon,
      color: 'mint',
      description: 'Machine learning models for automated dental assessment',
      achievements: [
        '95%+ accuracy in tooth segmentation',
        'Automated pathology detection',
        'Real-time quality assessment',
        'Multi-modal AI integration'
      ]
    },
    {
      title: 'Computer Vision Systems',
      icon: EyeIcon,
      color: 'dental',
      description: 'Advanced imaging and visual processing technologies',
      achievements: [
        'Multi-camera calibration systems',
        'Structured light optimization',
        'Real-time image processing',
        'Hardware-agnostic interfaces'
      ]
    },
    {
      title: 'Clinical Workflow Integration',
      icon: BeakerIcon,
      color: 'mint',
      description: 'Research into optimal dental scanning workflows',
      achievements: [
        'Evidence-based scanning protocols',
        'CAD/CAM workflow optimization',
        'Quality metrics standardization',
        'Clinical validation studies'
      ]
    }
  ]

  const publications = [
    {
      title: 'Real-Time TSDF Fusion for Intraoral 3D Reconstruction',
      authors: 'OpenDentalScan Research Team',
      journal: 'Journal of Dental Technology',
      year: '2024',
      type: 'Research Paper',
      abstract: 'Novel approach to real-time volumetric fusion achieving sub-millimeter accuracy in dental scanning applications.',
      metrics: { citations: 45, downloads: 1200 }
    },
    {
      title: 'AI-Driven Tooth Segmentation: A Comparative Study',
      authors: 'Dr. Sarah Chen, Dr. Michael Rodriguez',
      journal: 'International Conference on Medical AI',
      year: '2024',
      type: 'Conference Paper',
      abstract: 'Comprehensive evaluation of neural network architectures for automated tooth segmentation in 3D dental scans.',
      metrics: { citations: 32, downloads: 890 }
    },
    {
      title: 'Cost-Effective Dental Scanning: Technology Democratization',
      authors: 'OpenDentalScan Economics Team',
      journal: 'Healthcare Technology Review',
      year: '2023',
      type: 'Industry Report',
      abstract: 'Analysis of how modern open-source technologies enable affordable professional dental scanning solutions.',
      metrics: { citations: 28, downloads: 2100 }
    },
    {
      title: 'Multi-Modal Sensor Fusion in Dental Imaging',
      authors: 'Dr. Alex Kim, Dr. Jennifer Liu',
      journal: 'IEEE Transactions on Medical Imaging',
      year: '2023',
      type: 'Research Paper',
      abstract: 'Advanced sensor fusion techniques combining structured light, stereo vision, and depth sensing for enhanced dental reconstruction.',
      metrics: { citations: 67, downloads: 1500 }
    },
    {
      title: 'Clinical Validation of Open-Source Dental Scanning',
      authors: 'Dr. Robert Thompson, Clinical Research Team',
      journal: 'Journal of Digital Dentistry',
      year: '2023',
      type: 'Clinical Study',
      abstract: 'Multi-center clinical trial validating accuracy and reliability of cost-effective dental scanning technology.',
      metrics: { citations: 89, downloads: 3200 }
    }
  ]

  const innovations = [
    {
      title: 'GPU-Accelerated TSDF',
      description: 'Revolutionary volumetric fusion using modern GPU computing',
      impact: '10x faster processing',
      status: 'Implemented'
    },
    {
      title: 'Adaptive AI Models',
      description: 'Self-improving neural networks for dental analysis',
      impact: '15% accuracy improvement',
      status: 'In Development'
    },
    {
      title: 'Multi-Device Synchronization',
      description: 'Seamless integration across different scanning hardware',
      impact: 'Universal compatibility',
      status: 'Beta Testing'
    },
    {
      title: 'Real-Time Quality Metrics',
      description: 'Instant feedback on scan quality during acquisition',
      impact: '40% reduction in rescans',
      status: 'Implemented'
    }
  ]

  const collaborations = [
    {
      name: 'Stanford Medical AI Lab',
      type: 'Academic Partnership',
      focus: 'Advanced neural network architectures',
      duration: '2023-2025'
    },
    {
      name: 'MIT Computer Vision Group',
      type: 'Research Collaboration',
      focus: '3D reconstruction algorithms',
      duration: '2022-2024'
    },
    {
      name: 'Mayo Clinic Digital Health',
      type: 'Clinical Validation',
      focus: 'Real-world clinical testing',
      duration: '2023-2024'
    },
    {
      name: 'European Dental Research Consortium',
      type: 'International Partnership',
      focus: 'Standardization and protocols',
      duration: '2024-2026'
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
              <div className="inline-flex items-center space-x-2 bg-mint-100 text-mint-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
                <AcademicCapIcon className="w-4 h-4" />
                <span>Research & Innovation</span>
              </div>
              
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-neutral-900 mb-6">
                Advancing{' '}
                <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                  Dental Technology
                </span>
              </h1>
              
              <p className="text-xl text-neutral-600 max-w-3xl mx-auto mb-8">
                Pioneering research in 3D reconstruction, artificial intelligence, and computer vision 
                to revolutionize dental scanning technology and make it accessible worldwide.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Research Areas */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-16">
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">Research Focus Areas</h2>
              <p className="text-neutral-600 max-w-2xl mx-auto">
                Our multidisciplinary research spans computer science, engineering, and clinical dentistry
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-8">
              {researchAreas.map((area, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="bg-gradient-to-br from-white to-neutral-50 rounded-2xl p-8 border border-neutral-200 hover:border-dental-200 transition-all duration-300 hover:shadow-gentle"
                >
                  <div className={`w-14 h-14 rounded-xl mb-6 flex items-center justify-center ${
                    area.color === 'dental'
                      ? 'bg-gradient-to-br from-dental-500 to-dental-600'
                      : 'bg-gradient-to-br from-mint-500 to-mint-600'
                  } shadow-soft`}>
                    <area.icon className="w-7 h-7 text-white" />
                  </div>

                  <h3 className="text-xl font-bold text-neutral-900 mb-3">{area.title}</h3>
                  <p className="text-neutral-600 mb-6">{area.description}</p>

                  <div className="space-y-3">
                    {area.achievements.map((achievement, achIndex) => (
                      <div key={achIndex} className="flex items-center space-x-3">
                        <div className={`w-2 h-2 rounded-full ${
                          area.color === 'dental' ? 'bg-dental-400' : 'bg-mint-400'
                        }`} />
                        <span className="text-neutral-700 text-sm">{achievement}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Publications */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-16">
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">Research Publications</h2>
              <p className="text-neutral-600 max-w-2xl mx-auto">
                Peer-reviewed research advancing the field of dental technology
              </p>
            </motion.div>

            <div className="space-y-6">
              {publications.map((pub, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="bg-white rounded-2xl p-8 border border-neutral-200 hover:border-dental-200 transition-all duration-300 hover:shadow-gentle"
                >
                  <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-3">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                          pub.type === 'Research Paper' ? 'bg-dental-100 text-dental-700' :
                          pub.type === 'Conference Paper' ? 'bg-mint-100 text-mint-700' :
                          pub.type === 'Clinical Study' ? 'bg-blue-100 text-blue-700' :
                          'bg-neutral-100 text-neutral-700'
                        }`}>
                          {pub.type}
                        </span>
                        <span className="text-neutral-500 text-sm">{pub.year}</span>
                      </div>
                      
                      <h3 className="text-xl font-bold text-neutral-900 mb-2">{pub.title}</h3>
                      <p className="text-neutral-600 text-sm mb-3">{pub.authors}</p>
                      <p className="text-neutral-500 text-sm mb-4 italic">{pub.journal}</p>
                      <p className="text-neutral-700 leading-relaxed">{pub.abstract}</p>
                    </div>
                    
                    <div className="lg:text-right">
                      <div className="flex lg:flex-col gap-4 lg:gap-2">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-dental-600">{pub.metrics.citations}</div>
                          <div className="text-xs text-neutral-500">Citations</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-mint-600">{pub.metrics.downloads}</div>
                          <div className="text-xs text-neutral-500">Downloads</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Innovations */}
      <section className="py-16 bg-gradient-to-br from-neutral-900 to-dental-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-16">
              <h2 className="text-3xl font-bold mb-4">Current Innovations</h2>
              <p className="text-neutral-300 max-w-2xl mx-auto">
                Breakthrough technologies currently in development and deployment
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-8">
              {innovations.map((innovation, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20"
                >
                  <div className="flex items-start justify-between mb-4">
                    <h3 className="text-xl font-bold">{innovation.title}</h3>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      innovation.status === 'Implemented' ? 'bg-mint-500/20 text-mint-300' :
                      innovation.status === 'Beta Testing' ? 'bg-yellow-500/20 text-yellow-300' :
                      'bg-blue-500/20 text-blue-300'
                    }`}>
                      {innovation.status}
                    </span>
                  </div>
                  <p className="text-neutral-300 mb-4">{innovation.description}</p>
                  <div className="flex items-center space-x-2">
                    <TrophyIcon className="w-4 h-4 text-dental-300" />
                    <span className="text-dental-300 font-medium">{innovation.impact}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Collaborations */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants} className="text-center mb-16">
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">Research Collaborations</h2>
              <p className="text-neutral-600 max-w-2xl mx-auto">
                Partnerships with leading institutions advancing dental technology research
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-8">
              {collaborations.map((collab, index) => (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="bg-gradient-to-br from-dental-50 to-mint-50 rounded-2xl p-6 border border-dental-100"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-bold text-neutral-900">{collab.name}</h3>
                      <p className="text-dental-600 font-medium text-sm">{collab.type}</p>
                    </div>
                    <div className="text-right">
                      <CalendarIcon className="w-5 h-5 text-neutral-400 mb-1" />
                      <p className="text-neutral-500 text-sm">{collab.duration}</p>
                    </div>
                  </div>
                  <p className="text-neutral-700">{collab.focus}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div variants={itemVariants}>
              <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                Interested in Collaboration?
              </h2>
              <p className="text-xl text-neutral-600 mb-8">
                Join our research network and help advance the future of dental technology
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button className="bg-dental-600 hover:bg-dental-700 text-white px-8 py-4 rounded-xl font-semibold text-lg transition-colors duration-200 shadow-dental">
                  Research Partnership
                </button>
                <button className="bg-white hover:bg-neutral-50 text-dental-600 px-8 py-4 rounded-xl font-semibold text-lg border-2 border-dental-200 hover:border-dental-300 transition-all duration-200">
                  Academic Program
                </button>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default ResearchPage