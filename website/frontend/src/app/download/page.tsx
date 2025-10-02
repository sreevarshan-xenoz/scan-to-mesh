'use client'

import { useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import Link from 'next/link'
import { 
  ArrowDownTrayIcon,
  CheckCircleIcon,
  ComputerDesktopIcon,
  DevicePhoneMobileIcon,
  CpuChipIcon,
  CloudIcon,
  DocumentTextIcon,
  PlayCircleIcon,
  ShieldCheckIcon,
  ClockIcon,
  UserGroupIcon,
  CogIcon,
  ArrowRightIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowLeftIcon
} from '@heroicons/react/24/outline'

const downloadPackages = [
  {
    id: 'professional',
    name: 'OpenDentalScan Professional',
    version: '2.1.0',
    description: 'Complete dental scanning solution with AI analysis and cloud sync',
    price: '$499',
    license: 'Commercial License',
    platforms: ['Windows', 'macOS', 'Linux'],
    size: '2.4 GB',
    features: [
      'Real-time 3D scanning',
      'AI-powered analysis',
      'Cloud synchronization',
      'Multi-format export',
      'Professional support',
      'Regular updates'
    ],
    requirements: {
      os: 'Windows 10+, macOS 11+, Ubuntu 20.04+',
      cpu: 'Intel i5 8th gen or AMD Ryzen 5 3600',
      ram: '16 GB RAM',
      gpu: 'NVIDIA GTX 1060 or AMD RX 580',
      storage: '10 GB available space',
      camera: 'RGB-D camera (Intel RealSense, Azure Kinect)'
    },
    downloads: [
      { platform: 'Windows', file: 'OpenDentalScan-Pro-2.1.0-Windows.exe', size: '2.4 GB' },
      { platform: 'macOS', file: 'OpenDentalScan-Pro-2.1.0-macOS.dmg', size: '2.2 GB' },
      { platform: 'Linux', file: 'OpenDentalScan-Pro-2.1.0-Linux.AppImage', size: '2.3 GB' }
    ]
  },
  {
    id: 'community',
    name: 'OpenDentalScan Community',
    version: '2.1.0',
    description: 'Open-source dental scanning with core features',
    price: 'Free',
    license: 'MIT License',
    platforms: ['Windows', 'macOS', 'Linux'],
    size: '1.8 GB',
    features: [
      'Basic 3D scanning',
      'Standard export formats',
      'Community support',
      'Open source code',
      'Plugin system',
      'Educational use'
    ],
    requirements: {
      os: 'Windows 10+, macOS 11+, Ubuntu 18.04+',
      cpu: 'Intel i3 8th gen or AMD Ryzen 3 3200G',
      ram: '8 GB RAM',
      gpu: 'Integrated graphics or dedicated GPU',
      storage: '5 GB available space',
      camera: 'RGB-D camera or webcam'
    },
    downloads: [
      { platform: 'Windows', file: 'OpenDentalScan-Community-2.1.0-Windows.exe', size: '1.8 GB' },
      { platform: 'macOS', file: 'OpenDentalScan-Community-2.1.0-macOS.dmg', size: '1.7 GB' },
      { platform: 'Linux', file: 'OpenDentalScan-Community-2.1.0-Linux.AppImage', size: '1.8 GB' }
    ]
  },
  {
    id: 'mobile',
    name: 'OpenDentalScan Mobile',
    version: '1.5.0',
    description: 'Mobile scanning app for iOS and Android devices',
    price: '$99',
    license: 'Mobile License',
    platforms: ['iOS', 'Android'],
    size: '450 MB',
    features: [
      'Mobile 3D scanning',
      'LiDAR support (iOS)',
      'Cloud sync',
      'Basic analysis',
      'Share & export',
      'Offline mode'
    ],
    requirements: {
      os: 'iOS 14+ or Android 10+',
      cpu: 'A12 Bionic or Snapdragon 855',
      ram: '4 GB RAM',
      gpu: 'Integrated graphics',
      storage: '2 GB available space',
      camera: 'TrueDepth camera (iOS) or ToF sensor'
    },
    downloads: [
      { platform: 'iOS', file: 'App Store', size: '420 MB' },
      { platform: 'Android', file: 'Google Play', size: '450 MB' }
    ]
  },
  {
    id: 'sdk',
    name: 'OpenDentalScan SDK',
    version: '2.1.0',
    description: 'Developer toolkit for custom integrations',
    price: '$1,999',
    license: 'Developer License',
    platforms: ['Cross-platform'],
    size: '800 MB',
    features: [
      'C++ & Python APIs',
      'Custom algorithms',
      'Hardware integration',
      'Documentation',
      'Sample projects',
      'Technical support'
    ],
    requirements: {
      os: 'Windows, macOS, Linux',
      cpu: 'Modern multi-core processor',
      ram: '8 GB RAM',
      gpu: 'Optional GPU acceleration',
      storage: '5 GB available space',
      camera: 'Configurable camera support'
    },
    downloads: [
      { platform: 'All Platforms', file: 'OpenDentalScan-SDK-2.1.0.zip', size: '800 MB' }
    ]
  }
]

const additionalDownloads = [
  {
    name: 'Sample Data Pack',
    description: 'Example dental scans for testing and development',
    size: '1.2 GB',
    file: 'OpenDentalScan-SampleData-v2.zip',
    icon: DocumentTextIcon
  },
  {
    name: 'Video Tutorials',
    description: 'Complete video guide series (offline viewing)',
    size: '3.4 GB',
    file: 'OpenDentalScan-Tutorials-2024.zip',
    icon: PlayCircleIcon
  },
  {
    name: 'Hardware Drivers',
    description: 'Camera drivers and calibration tools',
    size: '245 MB',
    file: 'OpenDentalScan-Drivers-v2.1.exe',
    icon: CogIcon
  }
]

const DownloadPage = () => {
  const shouldReduceMotion = useReducedMotion()
  const [selectedPackage, setSelectedPackage] = useState(downloadPackages[0])
  const [selectedPlatform, setSelectedPlatform] = useState('Windows')

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

  const getDownloadForPlatform = (pkg: any, platform: string) => {
    return pkg.downloads.find((d: any) => d.platform === platform) || pkg.downloads[0]
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
                <ArrowDownTrayIcon className="w-4 h-4" />
                <span>Download Center</span>
              </div>
              
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-neutral-900 mb-6">
                Download{' '}
                <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                  OpenDentalScan
                </span>
              </h1>
              
              <p className="text-xl text-neutral-600 max-w-3xl mx-auto mb-8">
                Choose the right version for your needs. From professional dental practices 
                to developers and researchers - we have you covered.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Package Selection */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            {/* Package Cards */}
            <motion.div variants={itemVariants} className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
              {downloadPackages.map((pkg) => (
                <button
                  key={pkg.id}
                  onClick={() => setSelectedPackage(pkg)}
                  className={`text-left p-6 rounded-2xl border-2 transition-all duration-200 ${
                    selectedPackage.id === pkg.id
                      ? 'border-dental-500 bg-dental-50 shadow-lg'
                      : 'border-neutral-200 bg-white hover:border-dental-200 hover:shadow-md'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-bold text-neutral-900">{pkg.name}</h3>
                    <div className={`px-2 py-1 rounded text-xs font-medium ${
                      pkg.price === 'Free' 
                        ? 'bg-mint-100 text-mint-700'
                        : 'bg-dental-100 text-dental-700'
                    }`}>
                      {pkg.price}
                    </div>
                  </div>
                  
                  <p className="text-sm text-neutral-600 mb-4">{pkg.description}</p>
                  
                  <div className="flex items-center justify-between text-xs text-neutral-500">
                    <span>v{pkg.version}</span>
                    <span>{pkg.size}</span>
                  </div>
                </button>
              ))}
            </motion.div>

            {/* Selected Package Details */}
            <motion.div variants={itemVariants} className="bg-white rounded-3xl p-8 border border-neutral-200 shadow-gentle">
              <div className="grid lg:grid-cols-3 gap-8">
                
                {/* Package Info */}
                <div className="lg:col-span-2">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h2 className="text-3xl font-bold text-neutral-900 mb-2">{selectedPackage.name}</h2>
                      <p className="text-lg text-neutral-600 mb-4">{selectedPackage.description}</p>
                      <div className="flex items-center space-x-4 text-sm text-neutral-500">
                        <span>Version {selectedPackage.version}</span>
                        <span>•</span>
                        <span>{selectedPackage.license}</span>
                        <span>•</span>
                        <span>{selectedPackage.size}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold text-dental-600">{selectedPackage.price}</div>
                    </div>
                  </div>

                  {/* Features */}
                  <div className="mb-8">
                    <h3 className="text-xl font-bold text-neutral-900 mb-4">Features</h3>
                    <div className="grid md:grid-cols-2 gap-3">
                      {selectedPackage.features.map((feature, index) => (
                        <div key={index} className="flex items-center space-x-3">
                          <CheckCircleIcon className="w-5 h-5 text-mint-600 flex-shrink-0" />
                          <span className="text-neutral-700">{feature}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* System Requirements */}
                  <div>
                    <h3 className="text-xl font-bold text-neutral-900 mb-4">System Requirements</h3>
                    <div className="bg-neutral-50 rounded-xl p-6">
                      <div className="grid md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="font-medium text-neutral-900 mb-1">Operating System</div>
                          <div className="text-neutral-600">{selectedPackage.requirements.os}</div>
                        </div>
                        <div>
                          <div className="font-medium text-neutral-900 mb-1">Processor</div>
                          <div className="text-neutral-600">{selectedPackage.requirements.cpu}</div>
                        </div>
                        <div>
                          <div className="font-medium text-neutral-900 mb-1">Memory</div>
                          <div className="text-neutral-600">{selectedPackage.requirements.ram}</div>
                        </div>
                        <div>
                          <div className="font-medium text-neutral-900 mb-1">Graphics</div>
                          <div className="text-neutral-600">{selectedPackage.requirements.gpu}</div>
                        </div>
                        <div>
                          <div className="font-medium text-neutral-900 mb-1">Storage</div>
                          <div className="text-neutral-600">{selectedPackage.requirements.storage}</div>
                        </div>
                        <div>
                          <div className="font-medium text-neutral-900 mb-1">Camera</div>
                          <div className="text-neutral-600">{selectedPackage.requirements.camera}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Download Section */}
                <div>
                  <div className="bg-gradient-to-br from-dental-600 to-mint-500 rounded-2xl p-6 text-white mb-6">
                    <h3 className="text-xl font-bold mb-4">Download Now</h3>
                    
                    {/* Platform Selection */}
                    <div className="mb-6">
                      <div className="text-sm opacity-90 mb-2">Select Platform:</div>
                      <div className="space-y-2">
                        {selectedPackage.platforms.map((platform) => (
                          <button
                            key={platform}
                            onClick={() => setSelectedPlatform(platform)}
                            className={`w-full text-left px-3 py-2 rounded-lg transition-colors duration-200 ${
                              selectedPlatform === platform
                                ? 'bg-white/20 border border-white/30'
                                : 'bg-white/10 hover:bg-white/15'
                            }`}
                          >
                            <div className="flex items-center space-x-2">
                              {platform === 'iOS' || platform === 'Android' ? (
                                <DevicePhoneMobileIcon className="w-4 h-4" />
                              ) : (
                                <ComputerDesktopIcon className="w-4 h-4" />
                              )}
                              <span>{platform}</span>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Download Button */}
                    <button className="w-full bg-white text-dental-600 hover:bg-neutral-100 px-6 py-4 rounded-xl font-bold transition-colors duration-200 flex items-center justify-center space-x-2">
                      <ArrowDownTrayIcon className="w-5 h-5" />
                      <span>
                        Download for {selectedPlatform}
                      </span>
                    </button>
                    
                    <div className="text-center mt-3 text-sm opacity-75">
                      {getDownloadForPlatform(selectedPackage, selectedPlatform).size}
                    </div>
                  </div>

                  {/* Additional Info */}
                  <div className="space-y-4 text-sm">
                    <div className="flex items-start space-x-3 p-4 bg-mint-50 rounded-lg">
                      <ShieldCheckIcon className="w-5 h-5 text-mint-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-medium text-mint-900">Secure Download</div>
                        <div className="text-mint-700">All files are digitally signed and virus-free</div>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3 p-4 bg-dental-50 rounded-lg">
                      <ClockIcon className="w-5 h-5 text-dental-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-medium text-dental-900">Regular Updates</div>
                        <div className="text-dental-700">Automatic updates with new features</div>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3 p-4 bg-neutral-50 rounded-lg">
                      <UserGroupIcon className="w-5 h-5 text-neutral-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-medium text-neutral-900">Community Support</div>
                        <div className="text-neutral-700">Join our active user community</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Additional Downloads */}
      <section className="py-16 bg-neutral-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <motion.div variants={itemVariants} className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                Additional Downloads
              </h2>
              <p className="text-xl text-neutral-600 max-w-2xl mx-auto">
                Enhance your OpenDentalScan experience with these supplementary resources
              </p>
            </motion.div>

            <motion.div variants={itemVariants} className="grid md:grid-cols-3 gap-6">
              {additionalDownloads.map((item, index) => (
                <div key={index} className="bg-white p-6 rounded-2xl border border-neutral-200 shadow-soft">
                  <div className="flex items-center space-x-3 mb-4">
                    <item.icon className="w-8 h-8 text-dental-600" />
                    <h3 className="text-lg font-bold text-neutral-900">{item.name}</h3>
                  </div>
                  
                  <p className="text-neutral-600 mb-4">{item.description}</p>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-neutral-500">{item.size}</span>
                    <button className="flex items-center space-x-2 text-dental-600 hover:text-dental-700 font-medium transition-colors duration-200">
                      <ArrowDownTrayIcon className="w-4 h-4" />
                      <span>Download</span>
                    </button>
                  </div>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Installation Guide */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <motion.div variants={itemVariants} className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                Quick Installation Guide
              </h2>
              <p className="text-xl text-neutral-600">
                Get up and running in minutes with our step-by-step guide
              </p>
            </motion.div>

            <motion.div variants={itemVariants} className="bg-white rounded-3xl p-8 border border-neutral-200 shadow-gentle">
              <div className="space-y-8">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-dental-600 text-white rounded-full flex items-center justify-center font-bold">
                    1
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-neutral-900 mb-2">Download & Install</h3>
                    <p className="text-neutral-600">
                      Download the appropriate package for your platform and run the installer. 
                      Follow the on-screen instructions to complete the installation.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-dental-600 text-white rounded-full flex items-center justify-center font-bold">
                    2
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-neutral-900 mb-2">Connect Your Camera</h3>
                    <p className="text-neutral-600">
                      Connect your RGB-D camera (Intel RealSense, Azure Kinect, or compatible device) 
                      and install any required drivers from the Hardware Drivers package.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-dental-600 text-white rounded-full flex items-center justify-center font-bold">
                    3
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-neutral-900 mb-2">Calibrate & Test</h3>
                    <p className="text-neutral-600">
                      Run the calibration wizard to optimize your camera settings. 
                      Use the sample data pack to test scanning and export functionality.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-mint-600 text-white rounded-full flex items-center justify-center font-bold">
                    ✓
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-neutral-900 mb-2">Start Scanning</h3>
                    <p className="text-neutral-600">
                      You're ready to start scanning! Check out our video tutorials for 
                      best practices and advanced techniques.
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex items-start space-x-3">
                  <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium text-yellow-900">Need Help?</div>
                    <div className="text-yellow-800 text-sm">
                      Visit our <Link href="/docs" className="underline hover:no-underline">documentation</Link> or 
                      <Link href="/contact" className="underline hover:no-underline ml-1">contact support</Link> for assistance.
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <motion.div variants={itemVariants} className="bg-gradient-to-r from-dental-600 to-mint-500 rounded-3xl p-12 text-white">
              <h2 className="text-3xl md:text-4xl font-bold mb-6">
                Ready to Get Started?
              </h2>
              <p className="text-xl mb-8 opacity-90">
                Join thousands of dental professionals using OpenDentalScan worldwide
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  href="/demo"
                  className="bg-white text-dental-600 hover:bg-neutral-100 px-8 py-4 rounded-xl font-bold transition-colors duration-200"
                >
                  Try Demo First
                </Link>
                <Link
                  href="/docs/getting-started"
                  className="border-2 border-white text-white hover:bg-white hover:text-dental-600 px-8 py-4 rounded-xl font-bold transition-colors duration-200 flex items-center justify-center space-x-2"
                >
                  <span>View Documentation</span>
                  <ArrowRightIcon className="w-4 h-4" />
                </Link>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default DownloadPage