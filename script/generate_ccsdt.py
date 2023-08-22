import ase;
from ase.io.trajectory import Trajectory
from ase import io;
import os;

def write(filename,atoms):
    dic = {1:'H', 6:'C'};
    with open(filename,'w') as file:
        file.write('! CCSD(T) EOM-CCSD cc-pVTZ cc-pVTZ/C\n');
        file.write('! LargePrint PrintBasis KeepDens\n');
        file.write('%maxcore 3000\n');
#        file.write('%pal\n');
#        file.write('nprocs 32\n');
#        file.write('end\n');
        file.write('%MDCI\n');
        file.write('  nroots 3\n');
        file.write('  triples 1\n');
#        file.write(' MAXITER 500 \n');
#        file.write('  MaxDIIS 100\n');
#        file.write('  Lshift 0.01\n');
#        file.write('  density orbopt\n');
        file.write('END\n');

        file.write('%ELPROP\n');
        file.write('  dipole true\n');
        file.write('  quadrupole true\n');
        file.write('END\n');

#        file.write('%SCF\n');
#        file.write('  MAXITER 500\n');
##        file.write('  TolE 1e-6\n');
#        file.write('END\n');

        file.write('% OUTPUT\n');
        file.write('  Print[ P_Density ] 1        # converged density\n');
        file.write('  Print[ P_OneElec ] 1        # one electron matrix\n');
        file.write('  Print[ P_Overlap ] 1        # overlap matrix\n');
        file.write('  Print[ P_KinEn ] 1          # kinetic energy matrix\n');
        file.write('  Print[ P_Iter_F ] 1         # Fock matrix, for every iteration\n');
        file.write('  Print[ P_Fockian ] 1        # Fockian (not sure what this is)\n');
        file.write('END\n');

        file.write('* xyz 0 1 \n');
        pos = atoms.get_positions();
        atm = atoms.get_atomic_numbers();
        n = len(pos);
        for i in range(n):
            file.write(dic[atm[i]]+'\t');
            for j in range(3):
                file.write(str(pos[i][j])+'\t');
            file.write('\n');
        file.write('*');

l = os.listdir('./');
route = os.getcwd()+'/';
for fd in l:
#    if(fd[-3:]!='.py' and fd!='test'):
    if os.path.isdir(fd):
        traj = Trajectory(route+fd+'/data.traj');
        io.write(route+fd+'/XDATCAR', traj, format='vasp-xdatcar');

        steps = 1;
        i_index = 0;
        for atoms in traj:
            if(str(i_index) not in os.listdir(fd)):
                os.mkdir(route+fd+'/'+str(i_index));
            if(i_index%steps == 0):
                write(route+fd+'/'+str(i_index)+'/run.inp',atoms);

            i_index += 1;
        with open(route+fd+'/submit.sh','w') as file:
            file.write('#!/bin/bash\n');
            file.write('#SBATCH -o log-%j\n');
            file.write('#SBATCH -N 1\n');
            file.write('#SBATCH -n 48\n');
            file.write('#SBATCH --exclusive\n');
            file.write('#SBATCH -p xeon-p8\n\n');
            file.write('module load mpi/openmpi-4.1.1\n\n');
            file.write('for i in {0..499}\n');
            file.write('do\n    cd $i\n /home/gridsan/hxu1/apps/orca/orca  run.inp > log\n');
            file.write('    cd ../\n');
            file.write('done');
        print(fd);

        os.chdir(fd)
        os.system("sbatch submit.sh")
        os.chdir("../")

