"""Static CAG context: previous estimates injected into every prompt as few-shot examples.

All data is fictitious. Every example uses the same hourly rate (HOURLY_RATE_USD) and
the same layout, so the model learns one consistent way to estimate.
"""

HOURLY_RATE_USD = 60

ESTIMATION_EXAMPLES = [
    {
        "meeting_summary": (
            "The client needs a web platform to manage the inventory of three warehouses. "
            "Staff must add, edit and remove products, see stock levels per warehouse, and "
            "get an alert when stock is low. There are two user roles: admin and warehouse "
            "operator. Managers want a dashboard with the main metrics."
        ),
        "estimation": """## Estimate: Inventory Management Platform

### Task breakdown:
1. UI/UX design: 40 hours
2. Backend API (inventory CRUD, stock per warehouse, low-stock alerts): 60 hours
3. Authentication and roles (admin, operator): 20 hours
4. Dashboard with metrics: 30 hours
5. Testing and QA: 25 hours

**Total estimate: 175 hours**
**Estimated cost: 10,500 USD (175 h x 60 USD/h)**
**Recommended team: 2 full-stack developers + 1 UX designer (part-time)**
**Estimated duration: 6-8 weeks**
""",
    },
    {
        "meeting_summary": (
            "A consulting firm wants a new corporate website: five static pages, a blog that "
            "the marketing team can edit without a developer, and a contact form. Every form "
            "submission must create a lead in their existing CRM (Salesforce). The visual "
            "design is not done yet. They want basic SEO and Google Analytics."
        ),
        "estimation": """## Estimate: Corporate Website with Blog and CRM Integration

### Task breakdown:
1. UI/UX design (5 pages + blog templates): 24 hours
2. Frontend development (responsive pages): 40 hours
3. CMS setup and blog with editor: 24 hours
4. Contact form with validation and spam protection: 8 hours
5. CRM integration (Salesforce API, lead creation): 20 hours
6. SEO basics and Google Analytics: 8 hours
7. Testing and QA: 12 hours
8. Deployment and handover: 4 hours

**Total estimate: 140 hours**
**Estimated cost: 8,400 USD (140 h x 60 USD/h)**
**Recommended team: 1 full-stack developer + 1 designer (part-time)**
**Estimated duration: 4-5 weeks**
""",
    },
    {
        "meeting_summary": (
            "A chain of fitness studios needs a mobile app for iOS and Android. Members book "
            "and cancel classes, pay for class packs online, and receive reminders before each "
            "class. Studio staff need a simple web admin panel to create classes and see "
            "bookings."
        ),
        "estimation": """## Estimate: Class Booking Mobile App

### Task breakdown:
1. UI/UX design (mobile app + admin panel): 40 hours
2. Mobile app, iOS and Android (cross-platform): 120 hours
3. Backend API and web admin panel: 60 hours
4. Online payments (Stripe, class packs): 24 hours
5. Push notifications and reminders: 12 hours
6. Testing and QA (devices + payments): 30 hours
7. App Store and Google Play publishing: 8 hours

**Total estimate: 294 hours**
**Estimated cost: 17,640 USD (294 h x 60 USD/h)**
**Recommended team: 2 mobile developers + 1 backend developer + 1 UX designer (part-time)**
**Estimated duration: 10-12 weeks**
""",
    },
]
