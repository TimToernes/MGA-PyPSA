import pypsa

#adjust the path to pypsa examples directory
network = pypsa.Network(csv_folder_name="C:/Users/Tim/Anaconda3/Lib/site-packages/pypsa/examples/ac-dc-meshed/ac-dc-data")

#set to your favourite solver
solver_name = "glpk"

network.lopf(snapshots=network.snapshots,solver_name=solver_name)


print(network.generators.p_nom_opt)

print(network.generators_t.p)

print(network.storage_units.p_nom_opt)

print(network.storage_units_t.p)

print(network.lines.s_nom_opt)

print(network.lines_t.p0)