using SimplePDHG, LinearAlgebra, SparseArrays, MathOptInterface
using Test

@testset "SimplePDHG.jl tests" begin

    tol_default = 1e-4

    ## Dense LP example
    # =================

    @info "Testing for Dense LP example"
    @info "============================"

    # data matrix A 
    A = [0.490068149630612 0.026170943424868743 -0.4760197176111424 -0.2399289248736444 -0.14042229222117725 2.4109037066790364 -0.268377079801401 0.29679630468739243 0.458136730762251 1.27580114411093 -0.5819035580678447 -0.2958582683209307 -0.8689709322749227 -1.8268508405175734 -0.6913253849911165 1.3604011002798186 0.052511886606586206 -0.969428819906188 -0.8904569718650072 -0.17981947205601953; 1.3176014765163482 -0.35503596503452517 1.0442367423340218 -0.8598034961716939 -0.961225449964892 0.2199434630576387 -0.01887175453648225 -1.2368710493818948 1.24090454017234 1.184712915378006 2.0944622545003755 -0.551740735751168 0.5696176531854712 -0.5045732228071292 0.08426268763657509 1.4081311818149074 -1.0922427504947538 0.016262037994819244 -1.5739413897661128 -0.920055399543915; 0.34589080856636056 0.07616059448063386 -1.0986631459006249 1.9958563043320194 -0.45897679422274995 0.9052620350517069 -0.8672799566073214 0.288322873888227 1.5837928661117526 1.24383141428368 0.5362464633549305 -0.21960100162688156 -0.7717209509741461 0.2643558121750441 -1.037934422537801 0.4810327290075547 -0.1507772252715972 -0.8861058781326404 -0.42913255736622136 -1.543895052235031; 0.4292047753255398 0.7124016060403942 -0.11098049822764026 0.5068821181245639 0.27229434629254157 0.7990379917390127 -0.5404542806420642 0.17665994092773704 0.7104693618107027 0.07352025981898863 -0.01565098708681032 -1.0399986245899908 -2.1693670030962666 -0.20417822618184153 1.7088870854942086 1.5135277631666604 -0.14924735077754678 -0.2127788327702751 -0.3629129908577979 0.38614072177278336; -3.488121456481353 -0.8210018500516921 -1.331434189195498 -0.6235649595325159 0.35129982702837387 -0.6302127865486844 -0.31106107412892164 0.5677646258612373 0.7955983081990443 -0.45215570835811175 -0.22056898096943817 0.06477663455737091 -0.6625734536458293 0.6160173102067996 -0.5408201158354439 0.26074839934911065 -0.2679432278786877 0.6973342887400307 -0.4283743281423461 0.6348391278859632; 0.4994917967933287 0.916530342572851 0.05515497202852376 0.49478928302250547 -0.5933436069141836 -0.6886451059997338 0.21698713282118207 0.1412725413406646 2.1111565593632977 -1.9380473823948756 0.5033145786087198 0.2183107986116474 0.7289411810106307 -0.03513769069382656 1.6733918923380675 -0.4012083096252951 0.4045276617633314 -1.2637632528848426 0.7182929220850014 2.170859283311534; -0.17970519748017363 -0.6671990672519786 -0.0002766112002108874 0.35407319113772145 1.5063420367584277 -2.561746155679383 -0.1162581388105231 0.6225864155438156 -0.20954201845643458 -0.9716850337241267 0.05145405882151525 -1.5028107882657622 0.48124687878792516 -0.6387679754236566 -0.5126707656248445 -0.477176236970704 0.27135234726698204 -0.1256090206379606 0.25390841695288197 -0.058401428442787666; -1.0570240498287147 -0.7125169246984143 1.2159670790136914 0.4943181215097394 -1.2850065045226953 0.8311767481511261 0.7800295428228609 -0.006858124532381048 -0.9332632418724833 0.6778397157088871 -0.9011765132487847 -1.428606590858955 0.008621488881317169 0.18868479740600017 -0.504951348925174 -2.075754721461425 -0.33088433427341146 1.2127123118908045 1.1489244612143235 -1.4050120295733601; -0.6177893431994487 -0.14672292891689026 -0.5993414575078079 -0.9338114990181463 0.0648615210633744 1.1438656084432781 0.4057478753008923 -0.6713975038486499 -2.02762334194965 -1.059677286207581 -0.5236240064098991 0.6652603992476147 1.118249499978314 -0.1898289324977956 0.849920193573148 0.22955120403201265 0.9166756078266124 0.5046440499161858 0.3706135756920151 0.571258018080254; -0.17408179148072803 -0.9558465889581647 0.12861390126393005 -1.3914382882998928 -0.8866964321648728 -0.8452882834677059 -0.40255843330871044 1.0943087443583697 0.5592050450840116 0.377086234930535 1.5468831759339572 0.46799728663112505 0.09700480004331635 -0.569781469618532 -0.4390903185810493 -0.5250073847278554 -0.09350293125903092 -0.3594955726499566 -0.06025046162243457 -0.8662237329601492];

    # convert A to a sparse matrix

    A = sparse(A);

    # resource vector b

    b = [-0.02345452874422428, 0.4592924698471112, 0.5273456887141309, 0.1183342514085366, 0.38521501405221914, 0.06320663742271403, -0.29816072135673016, -0.12059707160163385, -0.3679307610842185, 0.3243480249745967];

    # cost vector c

    c = [0.16409188799833982, 0.5721382454747623, 0.16803928141786087, 0.07131334416428148, 0.2067070881262965, 0.13008260903577692, 0.11947941216975216, 0.2600514824019634, 0.12835385683498443, 0.12611211816599655, 0.2676983837907964, 0.1515711877118415, 0.23317742715157896, 0.10531940067876097, 0.3122667837804287, 0.1060828819902083, 0.26426565998991614, 0.2789245506495718, 0.019889101725802237, 0.14691296268795564];

    m, n = size(A)

    # create the data object
    problem = LP_Data(c, A, b, m, n)

    # create the setting data structure
    setting = PDHG_settings(maxit=100000, tol=tol_default, verbose=true, freq=10000)

    # solve the problem
    state, tol_final, _ = PDHG_solver(problem, setting)

    @test tol_final < 1e-4

    @info "Testing for Sparse LP example"
    @info "============================"

    ## Sparse LP example

    A = sparse([3, 10, 10, 2, 4, 9, 10, 2, 5, 6, 7, 10, 2, 8, 5, 7, 4, 3, 8, 1], [1, 4, 5, 6, 6, 8, 8, 9, 10, 10, 10, 12, 13, 13, 14, 14, 15, 17, 18, 20], [0.5156532999378246, 0.45293612002719896, 0.3979601946843593, 0.13705328111884874, 0.7124952361770818, 0.12115081554889895, 0.16389581591372748, 0.923714830241645, 0.4441327173512031, 0.5895350781459361, 0.36093103374823754, 0.38096010128800295, 0.0021506739545886777, 0.6721765889825463, 0.27633518638841903, 0.5793290520673758, 0.5730856157280215, 0.7297197241442733, 0.9815705363372546, 0.9514089244192082], 10, 20)

    b = [0.9076614668990857, 0.37020161526446355, 0.6603463336334293, 0.9518546928435478, 0.2870395088457023, 0.38085760220438586, 0.23341616150860003, 1.3213453977410774, 0.0833880493529622, 0.7453177779545501]

    c = [0.3139939520559397, 0.6068943938867927, 0.046970870731568026, 0.870576359061842, 0.7798349718972201, 0.4922347414740511, 0.6890536425088747, 0.017419893483798132, 0.28071662630833905, 0.31110043593776426, 0.2484620049437828, 0.4617027814105584, 0.6776720037848275, 0.5865891343104259, 0.8805289564904033, 0.2877910525059626, 0.325420848428544, 0.5131846829366469, 0.02345941928601447, 0.47984391369870294]

    m, n = size(A)

    # create the data object
    problem = LP_Data(c, A, b, m, n)

    3.1373320532369853

    # create the setting data structure
    setting = PDHG_settings(maxit=100000, tol=tol_default, verbose=true, freq=10000)

    # solve the problem
    state, tol_final = PDHG_solver(problem, setting)

    @test tol_final < 1e-4

end

# @testset "Sparse LP" begin

#     A = sparse([3, 10, 10, 2, 4, 9, 10, 2, 5, 6, 7, 10, 2, 8, 5, 7, 4, 3, 8, 1], [1, 4, 5, 6, 6, 8, 8, 9, 10, 10, 10, 12, 13, 13, 14, 14, 15, 17, 18, 20], [0.5156532999378246, 0.45293612002719896, 0.3979601946843593, 0.13705328111884874, 0.7124952361770818, 0.12115081554889895, 0.16389581591372748, 0.923714830241645, 0.4441327173512031, 0.5895350781459361, 0.36093103374823754, 0.38096010128800295, 0.0021506739545886777, 0.6721765889825463, 0.27633518638841903, 0.5793290520673758, 0.5730856157280215, 0.7297197241442733, 0.9815705363372546, 0.9514089244192082], 10, 20)

#     b = [0.9076614668990857, 0.37020161526446355, 0.6603463336334293, 0.9518546928435478, 0.2870395088457023, 0.38085760220438586, 0.23341616150860003, 1.3213453977410774, 0.0833880493529622, 0.7453177779545501]

#     c = [0.3139939520559397, 0.6068943938867927, 0.046970870731568026, 0.870576359061842, 0.7798349718972201, 0.4922347414740511, 0.6890536425088747, 0.017419893483798132, 0.28071662630833905, 0.31110043593776426, 0.2484620049437828, 0.4617027814105584, 0.6776720037848275, 0.5865891343104259, 0.8805289564904033, 0.2877910525059626, 0.325420848428544, 0.5131846829366469, 0.02345941928601447, 0.47984391369870294]

#     m, n = size(A)

#     # create the data object
#     problem = LP_Data(c, A, b, m, n)

#     3.1373320532369853

#     # create the setting data structure
#     setting = PDHG_settings(maxit=100000, tol=1e-6, verbose=true, freq=100)

#     # solve the problem
#     state, tol_final = PDHG_solver(problem, setting)

#     # get the optimal value
#     obj_value_PDHG = problem.c'*state.x

#     p_star = 3.1373320532369853 # the optimal value

#     @test abs(obj_value_PDHG - p_star) < 1e-4


# end
