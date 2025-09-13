export const mockData = {
  documents: [
    {
      id: 1,
      filename: "Company_Handbook_2024.pdf",
      upload_time: "2024-03-15T10:30:00Z",
      chunk_count: 42,
      status: "completed"
    },
    {
      id: 2,
      filename: "Technical_Documentation.pdf",
      upload_time: "2024-03-14T15:20:00Z",
      chunk_count: 28,
      status: "completed"
    },
    {
      id: 3,
      filename: "Project_Requirements.pdf",
      upload_time: "2024-03-13T09:45:00Z",
      chunk_count: 15,
      status: "completed"
    },
    {
      id: 4,
      filename: "Research_Paper_2024.pdf",
      upload_time: "2024-03-12T14:10:00Z",
      chunk_count: 67,
      status: "processing"
    }
  ],
  
  sampleAnswers: [
    "Based on the uploaded documents, here's what I found:\n\nThe company handbook outlines the key policies and procedures that all employees must follow. This includes information about work hours, vacation policies, and performance evaluation criteria.\n\nThe technical documentation provides detailed implementation guidelines for the current project architecture.",
    
    "According to the project requirements document:\n\n1. The system must support real-time data processing\n2. User authentication is required for all endpoints\n3. The application should be scalable to handle 10,000+ concurrent users\n4. All data must be encrypted in transit and at rest\n\nThese requirements form the foundation of our development approach.",
    
    "The research paper discusses innovative approaches to document processing and natural language understanding. Key findings include:\n\n• Machine learning models show 95% accuracy in document classification\n• Processing time has been reduced by 40% compared to traditional methods\n• User satisfaction scores increased significantly with the new interface",
    
    "From the technical documentation, the recommended architecture includes:\n\n- Frontend: React with TypeScript\n- Backend: FastAPI with Python\n- Database: PostgreSQL with Redis for caching\n- Authentication: JWT tokens with refresh mechanism\n- Deployment: Docker containers with Kubernetes orchestration\n\nThis stack ensures both performance and maintainability.",
    
    "The company handbook specifies the following vacation policy:\n\n• New employees: 15 days per year\n• 2-5 years experience: 20 days per year\n• 5+ years experience: 25 days per year\n• Additional personal days: 5 per year\n• Sick leave: Unlimited with manager approval\n\nAll vacation requests must be submitted at least 2 weeks in advance through the HR portal."
  ]
};