import chromadb

client = chromadb.PersistentClient(path="./chromaDB")

collection = client.create_collection(name="office_policies")

collection.add(
    # 1. Documents: The actual text content containing the company rules and guidelines
    documents=[
        "Leave Policy: Employees are entitled to 18 days of paid casual leave and 12 days of medical leave per calendar year. Leave requests must be submitted through the HR portal at least 3 business days in advance for approval.",
        
        "Office Timings & Core Hours: Standard office hours are 9:00 AM to 6:00 PM, Monday through Friday. Core collaboration hours are 11:00 AM to 4:00 PM, during which all team members are expected to be available for sync ups.",
        
        "Expense Reimbursement: Travel and client-entertainment expenses must be filed by the 25th of each month. All claims require original digital receipts attached to the expensify dashboard; claims without invoices will be rejected.",
        
        "IT Security & Passwords: Passwords for all internal systems (Git, AWS, Jira) must be changed every 90 days. Multi-Factor Authentication (MFA) is strictly mandatory for all active company accounts and staging environments.",
        
        "Work From Home (WFH) Guidelines: Employees can request up to 2 remote work days per week with prior manager approval. High-speed internet (minimum 50 Mbps) and a quiet working space are mandatory requirements for WFH days."
    ],
    
    # 2. Metadatas: Structured tags to enable precise filtering later
    metadatas=[
        {"department": "hr", "priority": "high", "audience": "all_employees"},
        {"department": "operations", "priority": "medium", "audience": "all_employees"},
        {"department": "finance", "priority": "high", "audience": "all_employees"},
        {"department": "it_security", "priority": "critical", "audience": "developers_and_staff"},
        {"department": "hr", "priority": "medium", "audience": "remote_and_hybrid"}
    ],
    
    # 3. IDs: Unique alpha-numeric strings to identify each document chunk
    ids=[
        "doc_hr_leave_001",
        "doc_ops_timing_002",
        "doc_fin_expense_003",
        "doc_it_secure_004",
        "doc_hr_wfh_005"
    ]
)

print("Data added to collection successfully!")