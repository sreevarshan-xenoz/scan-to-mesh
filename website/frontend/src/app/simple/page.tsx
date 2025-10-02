export default function SimplePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-dental-50 via-white to-mint-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-neutral-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-dental-500 to-mint-500 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-dental-700">OpenDentalScan</h1>
                <p className="text-xs text-neutral-500 -mt-1">Open Source</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="bg-dental-600 hover:bg-dental-700 text-white px-4 py-2 rounded-lg font-medium">
                Download
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="inline-flex items-center space-x-2 bg-dental-100 text-dental-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
              <span>Professional Grade • Advanced Technology</span>
            </div>
            
            <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
              Professional{' '}
              <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                Dental Scanning
              </span>{' '}
              Made Accessible
            </h1>
            
            <p className="text-xl text-neutral-600 mb-8 max-w-3xl mx-auto">
              Advanced dental scanner delivering professional-grade 3D reconstruction, 
              AI-powered analysis, and clinical workflow integration using modern open-source technologies.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="bg-dental-600 hover:bg-dental-700 text-white px-8 py-4 rounded-xl font-semibold text-lg">
                Get Started Free
              </button>
              <button className="bg-white hover:bg-neutral-50 text-dental-600 px-8 py-4 rounded-xl font-semibold text-lg border-2 border-dental-200">
                Watch Demo
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
              Professional Features
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Everything you need for professional dental scanning
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-white rounded-2xl p-8 border border-neutral-200 shadow-lg">
              <div className="w-14 h-14 bg-gradient-to-br from-dental-500 to-dental-600 rounded-xl mb-6 flex items-center justify-center">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-neutral-900 mb-4">Real-Time 3D Reconstruction</h3>
              <p className="text-neutral-600">GPU-accelerated TSDF fusion delivers professional-grade 3D models with sub-millimeter accuracy at 30 FPS.</p>
            </div>
            
            <div className="bg-white rounded-2xl p-8 border border-neutral-200 shadow-lg">
              <div className="w-14 h-14 bg-gradient-to-br from-mint-500 to-mint-600 rounded-xl mb-6 flex items-center justify-center">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-neutral-900 mb-4">AI-Powered Analysis</h3>
              <p className="text-neutral-600">Advanced neural networks for tooth segmentation, pathology detection, and automated dental numbering.</p>
            </div>
            
            <div className="bg-white rounded-2xl p-8 border border-neutral-200 shadow-lg">
              <div className="w-14 h-14 bg-gradient-to-br from-dental-500 to-dental-600 rounded-xl mb-6 flex items-center justify-center">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-neutral-900 mb-4">Modern Technology</h3>
              <p className="text-neutral-600">Built with cutting-edge open-source technologies for reliability, performance, and cost-effectiveness.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-neutral-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-dental-400 to-mint-400 rounded-xl flex items-center justify-center">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-bold">OpenDentalScan</h3>
                <p className="text-sm text-neutral-300">Professional</p>
              </div>
            </div>
            <p className="text-neutral-300 mb-4">
              Professional-grade dental scanning made accessible through modern technology innovation.
            </p>
            <p className="text-neutral-400 text-sm">
              © 2024 OpenDentalScan. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}