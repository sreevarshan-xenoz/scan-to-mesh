'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { 
  CpuChipIcon,
  CameraIcon,
  CircleStackIcon,
  CloudIcon,
  CubeIcon,
  ShieldCheckIcon,
  BeakerIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline'

const TechSpecs = () => {
  const shouldReduceMotion = useReducedMotion()

  const techCategories = [
    {
      title: '3D Processing Engine',
      icon: CubeIcon,
      color: 'dental',
      specs: [
        { label: 'TSDF Fusion', value: 'GPU-accelerated PyTorch' },
        { label: 'Voxel Resolution', value: '0.5-2mm configurable' },
        { label: 'Processing Speed', value: '30+ FPS real-time' },
        { label: 'Accuracy', value: 'Sub-millimeter precision' },
        { label: 'Volume Size', value: 'Up to 20cm³' },
        { label: 'Mesh Quality', value: 'Professional CAD/CAM grade' }
      ]
    },
    {
      title: 'AI/ML Capabilities',
      icon: CpuChipIcon,
      color: 'mint',
      specs: [
        { label: 'Model Format', value: 'ONNX Runtime + PyTorch' },
        { label: 'Segmentation Accuracy', value: '95%+ tooth detection' },
        { label: 'Inference Speed', value: '<100ms latency' },
        { label: 'Model Count', value: '22 specialized models' },
        { label: 'GPU Acceleration', value: 'CUDA + TensorRT support' },
        { label: 'Custom Training', value: 'Full pipeline included' }
      ]
    },
    {
      title: 'Hardware Support',
      icon: CameraIcon,
      color: 'dental',
      specs: [
        { label: 'Intel RealSense', value: 'D435i, L515, D455' },
        { label: 'Stereo Cameras', value: 'USB 3.0 stereo pairs' },
        { label: 'Structured Light', value: 'Custom projector systems' },
        { label: 'Webcams', value: 'Standard USB cameras' },
        { label: 'Resolution', value: 'Up to 1920x1080@30fps' },
        { label: 'Calibration', value: 'Automatic + manual tools' }
      ]
    },
    {
      title: 'System Requirements',
      icon: CircleStackIcon,
      color: 'mint',
      specs: [
        { label: 'OS Support', value: 'Windows, Linux, macOS' },
        { label: 'GPU', value: 'NVIDIA GTX 1060+ (8GB+ VRAM)' },
        { label: 'CPU', value: 'Intel i5-8400 / AMD Ryzen 5 2600+' },
        { label: 'RAM', value: '8GB minimum, 16GB recommended' },
        { label: 'Storage', value: '10GB+ SSD recommended' },
        { label: 'Python', value: '3.8+ with CUDA support' }
      ]
    },
    {
      title: 'Export & Integration',
      icon: CloudIcon,
      color: 'dental',
      specs: [
        { label: 'File Formats', value: 'STL, OBJ, PLY, DICOM' },
        { label: 'CAD/CAM', value: 'Direct workflow integration' },
        { label: 'Cloud Sync', value: 'Optional secure backup' },
        { label: 'API Access', value: 'RESTful + Python SDK' },
        { label: 'Database', value: 'SQLite + PostgreSQL' },
        { label: 'Reports', value: 'PDF clinical reports' }
      ]
    },
    {
      title: 'Security & Compliance',
      icon: ShieldCheckIcon,
      color: 'mint',
      specs: [
        { label: 'Data Privacy', value: 'HIPAA + GDPR compliant' },
        { label: 'Encryption', value: 'AES-256 at rest & transit' },
        { label: 'Access Control', value: 'Role-based permissions' },
        { label: 'Audit Trail', value: 'Complete activity logging' },
        { label: 'Local Processing', value: 'No cloud dependency' },
        { label: 'Open Source', value: 'Transparent security model' }
      ]
    }
  ]

  const performanceMetrics = [
    { metric: 'Frame Rate', value: '30+ FPS', description: 'Real-time processing' },
    { metric: 'Latency', value: '<100ms', description: 'AI inference speed' },
    { metric: 'Accuracy', value: '0.1mm', description: '3D reconstruction precision' },
    { metric: 'Cost Savings', value: '100x', description: 'vs commercial solutions' }
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
    <section id="specs" className="section-padding bg-white">
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
              <span>Technical Specifications</span>
            </div>
            
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-neutral-900 mb-6">
              Professional-Grade{' '}
              <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                Technical Foundation
              </span>
            </h2>
            
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
              Built on modern technologies and optimized for professional dental workflows. 
              Every component designed for reliability, performance, and clinical precision.
            </p>
          </motion.div>

          {/* Performance Metrics */}
          <motion.div variants={itemVariants} className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
            {performanceMetrics.map((metric, index) => (
              <div key={index} className="text-center bg-gradient-to-br from-dental-50 to-mint-50 rounded-2xl p-6 border border-dental-100">
                <div className="text-3xl md:text-4xl font-bold text-dental-600 mb-2">
                  {metric.value}
                </div>
                <div className="text-lg font-semibold text-neutral-900 mb-1">
                  {metric.metric}
                </div>
                <div className="text-sm text-neutral-600">
                  {metric.description}
                </div>
              </div>
            ))}
          </motion.div>

          {/* Technical Specifications Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {techCategories.map((category, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                className="bg-gradient-to-br from-white to-neutral-50 rounded-2xl p-8 border border-neutral-200 hover:border-dental-200 transition-all duration-300 hover:shadow-gentle"
              >
                {/* Category Header */}
                <div className="flex items-center space-x-4 mb-6">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                    category.color === 'dental' 
                      ? 'bg-gradient-to-br from-dental-500 to-dental-600' 
                      : 'bg-gradient-to-br from-mint-500 to-mint-600'
                  } shadow-soft`}>
                    <category.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-neutral-900">
                    {category.title}
                  </h3>
                </div>

                {/* Specifications List */}
                <div className="space-y-4">
                  {category.specs.map((spec, specIndex) => (
                    <div key={specIndex} className="flex justify-between items-start">
                      <div className="text-neutral-600 text-sm font-medium flex-1 mr-4">
                        {spec.label}
                      </div>
                      <div className={`text-sm font-semibold text-right ${
                        category.color === 'dental' ? 'text-dental-700' : 'text-mint-700'
                      }`}>
                        {spec.value}
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>

          {/* Architecture Diagram */}
          <motion.div variants={itemVariants} className="mt-16">
            <div className="bg-gradient-to-br from-neutral-900 to-dental-900 rounded-3xl p-8 text-white">
              <h3 className="text-2xl font-bold mb-8 text-center">System Architecture</h3>
              
              <div className="grid md:grid-cols-4 gap-6">
                {/* Hardware Layer */}
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-dental-400 to-dental-500 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                    <CameraIcon className="w-8 h-8" />
                  </div>
                  <h4 className="font-semibold mb-2">Hardware Layer</h4>
                  <p className="text-sm text-neutral-300">Cameras, Sensors, Calibration</p>
                </div>

                {/* Processing Layer */}
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-mint-400 to-mint-500 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                    <CpuChipIcon className="w-8 h-8" />
                  </div>
                  <h4 className="font-semibold mb-2">Processing Layer</h4>
                  <p className="text-sm text-neutral-300">TSDF Fusion, SLAM, AI Models</p>
                </div>

                {/* Service Layer */}
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-dental-400 to-mint-400 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                    <CircleStackIcon className="w-8 h-8" />
                  </div>
                  <h4 className="font-semibold mb-2">Service Layer</h4>
                  <p className="text-sm text-neutral-300">APIs, Database, Communication</p>
                </div>

                {/* Application Layer */}
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-mint-400 to-dental-400 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                    <ChartBarIcon className="w-8 h-8" />
                  </div>
                  <h4 className="font-semibold mb-2">Application Layer</h4>
                  <p className="text-sm text-neutral-300">UI, Workflow, Export</p>
                </div>
              </div>

              {/* Data Flow Arrows */}
              <div className="hidden md:flex justify-center items-center mt-8 space-x-8">
                {[...Array(3)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="w-8 h-0.5 bg-gradient-to-r from-dental-400 to-mint-400"
                    animate={shouldReduceMotion ? {} : {
                      opacity: [0.5, 1, 0.5],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: i * 0.5,
                    }}
                  />
                ))}
              </div>
            </div>
          </motion.div>

          {/* Bottom CTA */}
          <motion.div variants={itemVariants} className="text-center mt-16">
            <div className="bg-gradient-to-r from-dental-50 to-mint-50 rounded-2xl p-8 border border-dental-100">
              <h3 className="text-2xl font-bold text-neutral-900 mb-4">
                Ready to Explore the Technical Details?
              </h3>
              <p className="text-neutral-600 mb-6 max-w-2xl mx-auto">
                Dive deeper into our technical documentation, API references, and implementation guides.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <motion.button
                  className="flex items-center space-x-2 bg-dental-600 hover:bg-dental-700 text-white px-6 py-3 rounded-xl font-semibold transition-colors duration-200 shadow-dental"
                  whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                  whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                >
                  <BeakerIcon className="w-5 h-5" />
                  <span>Technical Documentation</span>
                </motion.button>
                
                <motion.button
                  className="flex items-center space-x-2 bg-white hover:bg-neutral-50 text-dental-600 px-6 py-3 rounded-xl font-semibold border border-dental-200 hover:border-dental-300 transition-all duration-200"
                  whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                  whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                >
                  <ChartBarIcon className="w-5 h-5" />
                  <span>API Reference</span>
                </motion.button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

export default TechSpecs