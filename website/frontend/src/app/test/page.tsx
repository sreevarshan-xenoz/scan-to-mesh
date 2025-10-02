export default function TestPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-green-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-blue-600 mb-8">
          OpenDentalScan Test Page
        </h1>
        
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Tailwind CSS Test
          </h2>
          <p className="text-gray-600 mb-4">
            If you can see this styled properly, Tailwind CSS is working correctly.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-100 p-4 rounded-lg">
              <h3 className="font-semibold text-blue-800">Blue Card</h3>
              <p className="text-blue-600">This should be blue themed</p>
            </div>
            
            <div className="bg-green-100 p-4 rounded-lg">
              <h3 className="font-semibold text-green-800">Green Card</h3>
              <p className="text-green-600">This should be green themed</p>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800">Gray Card</h3>
              <p className="text-gray-600">This should be gray themed</p>
            </div>
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-blue-500 to-green-500 text-white p-6 rounded-lg">
          <h2 className="text-xl font-bold mb-2">Gradient Test</h2>
          <p>This should have a blue to green gradient background.</p>
        </div>
      </div>
    </div>
  )
}