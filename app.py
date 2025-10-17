{
  "case_insensitive": true,
  "metrics": [
    {
      "label": "Missing food",
      "patterns": ["Missing Item (Food)"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Order wrong",
      "patterns": ["Order Wrong"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Missing condiments",
      "patterns": ["Missing Condiments"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Out of menu item",
      "patterns": ["Out Of Menu Item"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Missing bev",
      "patterns": ["Missing Item (Bev)"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Missing ingredients",
      "patterns": ["Missing Ingredient (Food)"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Packaging to-go complaint",
      "patterns": ["Packaging To Go Complaint"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },

    {
      "label": "Unprofessional/Unfriendly",
      "patterns": ["Unprofessional Behavior", "Unfriendly Attitude"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager directly involved",
      "patterns": ["Manager Directly Involved In Complaint"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager not available",
      "patterns": ["Management Not Available"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager did not visit",
      "patterns": ["Manager Did Not Visit"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Negative mgr-employee exchange",
      "patterns": ["Negative Manager-Employee Interaction"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager did not follow up",
      "patterns": ["Manager Did Not Follow Up"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Argued with guest",
      "patterns": ["Argued With Guest"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },

    {
      "label": "Long hold/no answer",
      "patterns": ["Long Hold/No Answer/Hung Up"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "No/insufficient compensation offered",
      "patterns": ["No/Unsatisfactory Compensation Offered By Restaurant"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Did not attempt to resolve",
      "patterns": ["Did Not Attempt To Resolve Issue"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Guest left without ordering",
      "patterns": ["Guest Left Without Dining or Ordering"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Unknowledgeable",
      "patterns": ["Unknowledgeable"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Did not open on time",
      "patterns": ["Didn[’']t Open/close On Time"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "No/poor apology",
      "patterns": ["No/Poor Apology"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    }
  ]
}
