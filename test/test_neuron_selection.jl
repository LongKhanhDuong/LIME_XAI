using ExplainableAI: MaxActivationNS, IndexNS

A = [-2.1694243, 2.4023275, 0.99464744, -0.1514646, 1.0307171]
ns1 = @inferred MaxActivationNS()
ns2 = @inferred IndexNS(4)

@test ns1(A) == 2
@test ns2(A) == 4