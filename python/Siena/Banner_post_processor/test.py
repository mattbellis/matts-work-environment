import numpy as np

################################################################################
def upper_level(courses):

    two_of_three = 0
    other_upper_level = 0
    for course in courses:
        if course=="PHYS310" or course=="PHYS410" or course=="PHYS400":
            two_of_three += 1
        elif course[0:5]=="PHYS400":
            other_upper_level += 1

    if two_of_three>=2 and two_of_three+other_upper_level>=4:
        return True
    else:
        print("Doesn't pass upper-level requirements!")
        return False
################################################################################


courses_taken = ["PHYS130",
                 "PHYS140",
                 "PHYS220",
                 "PHYS260",
                 "PHYS310",
                 ]


requirements = [["PHYS110", "PHYS130"], 
                ["PHYS120", "PHYS140"], 
                ["PHYS220"],
                upper_level,
                ]

for requirement in requirements:

    passed_req = False

    if type(requirement) is list:

        for req_course in requirement:

            if req_course in courses_taken:
                passed_req = True

    elif callable(requirement):

        passed_req = requirement(courses_taken)

    print(requirement, passed_req)
















