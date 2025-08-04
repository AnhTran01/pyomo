#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import os
import shutil
import subprocess
import datetime
from io import StringIO
from typing import Mapping, Optional, Sequence
from tempfile import mkdtemp
import sys 

from pyomo.common import Executable
from pyomo.common.dependencies import pathlib
from pyomo.common.config import ConfigValue, document_kwargs_from_configdict, ConfigDict
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.opt.results import (
    SolverResults,
    SolverStatus,
    Solution,
    # SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.results import (
    legacy_solution_status_map,
    legacy_termination_condition_map,
    legacy_solver_status_map,
    Results,
    SolutionStatus,
    # TerminationCondition,
)  
from pyomo.contrib.solver.solvers.gms_sol_reader import GMSSolutionLoader
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoSolutionError,
)

import math
import pyomo.core.base.suffix
from pyomo.opt.base.solvers import _extract_version
from pyomo.common.tee import TeeStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix
from pyomo.common.collections import ComponentMap

from pyomo.repn.plugins.gams_writer_v2 import GAMSWriterInfo, GAMSWriter
from pyomo.solvers.amplfunc_merge import amplfunc_merge

logger = logging.getLogger(__name__)

from pyomo.common.dependencies import attempt_import
import struct

def _gams_importer():
    try:
        import gams.core.gdx as gdx
        return gdx
    except ImportError:
        try:
            # fall back to the pre-GAMS-45.0 API
            import gdxcc
            return gdxcc
        except:
            # suppress the error from the old API and reraise the current API import error
            pass
        raise

gdxcc, gdxcc_available = attempt_import('gdxcc', importer=_gams_importer)

class GAMSConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.executable : Executable = self.declare(
            'executable',
            ConfigValue(
                default=Executable('gams'),
                description="Executable for gams. Defaults to searching the "
                "``PATH`` for the first available ``gams``.",
            ),
        )
        self.logfile : ConfigDict = self.declare(
            'logfile',
            ConfigValue(
                default=None,
                description="Filename to output GAMS log to a file.",
            ),
        )
        self.writer_config: ConfigDict = self.declare(
            'writer_config', GAMSWriter.CONFIG()
        )

# TODO: Fill in the description and doc strings for the GAMSResults class
class GAMSResults(Results):
    def __init__(self):
        super().__init__()
        self.return_code : ConfigDict = self.declare(
            'return_code',
            ConfigValue(
                default=None,
                description="Return code from the GAMS solver.",
            ),
        )
        self.user_time : ConfigDict = self.declare(
            'user_time',
            ConfigValue(
                default=None,
                description="The elapsed time it took to execute a solve statement in total.",
            ),
        )
        self.system_time : ConfigDict = self.declare(
            'system_time',
            ConfigValue(
                default=None,
                description="",
            ),
        )
        # self.termination_message : ConfigDict = self.declare(
        #     'termination_message',
        #     ConfigValue(
        #         default=None,
        #         description="",
        #     ),
        # )
        # self.message : ConfigDict = self.declare(
        #     'message',
        #     ConfigValue(
        #         default=None,
        #         description="Return message in case GAMS encounters an error.",
        #     ),
        # )
        # self.name : ConfigDict = self.declare(
        #     'name',
        #     ConfigValue(
        #         default=None,
        #         description="Output filename.",
        #     ),
        # )
        # self.lower_bound : ConfigDict = self.declare(
        #     'lower_bound',
        #     ConfigValue(
        #         default=None,
        #         description="Solution lowerbound.",
        #     ),
        # )
        # self.upper_bound : ConfigDict = self.declare(
        #     'upper_bound',
        #     ConfigValue(
        #         default=None,
        #         description="Solution upperbound.",
        #     ),
        # )
        # self.number_of_variables : ConfigDict = self.declare(
        #     'number_of_variables',
        #     ConfigValue(
        #         default=None,
        #         description="Number of variables in the model.",
        #     ),
        # )     
        # self.number_of_constraints : ConfigDict = self.declare(
        #     'number_of_constraints',
        #     ConfigValue(
        #         default=None,
        #         description="Number of constraints in the model.",
        #     ),
        # )          
        # self.number_of_nonzeros : ConfigDict = self.declare(
        #     'number_of_nonzeros',
        #     ConfigValue(
        #         default=None,
        #         description="Number of nonzeros in the model.",
        #     ),
        # )          
        # self.number_of_binary_variables : ConfigDict = self.declare(
        #     'number_of_binary_variables',
        #     ConfigValue(
        #         default=None,
        #         description="Number of binary variable in the model.",
        #     ),
        # )      
        # self.number_of_integer_variables : ConfigDict = self.declare(
        #     'number_of_integer_variables',
        #     ConfigValue(
        #         default=None,
        #         description="Number of integer variable in the model.",
        #     ),
        # )       
        # self.number_of_continuous_variables : ConfigDict = self.declare(
        #     'number_of_continuous_variables',
        #     ConfigValue(
        #         default=None,
        #         description="Number of continuous in the model.",
        #     ),
        # )              
        # self.number_of_objectives : ConfigDict = self.declare(
        #     'number_of_objectives',
        #     ConfigValue(
        #         default=None,
        #         description="Number of objective function in the model.",
        #     ),
        # )        
        self.gams_termination_condition : ConfigDict = self.declare(
            'gams_termination_condition',
            ConfigValue(
                default=None,
                description="Include additional TerminationCondition domain."
            ),
        )        
        self.gams_solver_status : ConfigDict = self.declare(
            'gams_solver_status',
            ConfigValue(
                default=None,
                description="Include additional SolverStatus domain."
            ),
        )      
        # self.soln : Solution = self.declare(
        #     'soln',
        #     ConfigValue(
        #         default=None,
        #         description="Hold reference to the Solution() class."
        #     ),
        # ) 

class GAMS(SolverBase):
    CONFIG = GAMSConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._writer = GAMSWriter()
        self._available_cache = None
        self._version_cache = None

    def available(self, config=None, exception_flag=True):
        if config is None:
            config = self.config

        """True if the solver is available."""
        exe = config.executable

        if not exe.available():
            if not exception_flag:
                return False
            raise NameError(
                "No 'gams' command found on system PATH - GAMS shell "
                "solver functionality is not available."
            )
        # New versions of GAMS require a license to run anything.
        # Instead of parsing the output, we will try solving a trivial
        # model.
        avail = self._run_simple_model(config, 1)
        if not avail and exception_flag:
            raise NameError(
                "'gams' command failed to solve a simple model - "
                "GAMS solver functionality is not available."
            )
        return avail

    def _run_simple_model(self, config, n):
        solver_exec = config.executable.path()
        if solver_exec is None:
            return False
        tmpdir = mkdtemp()
        try:
            test = os.path.join(tmpdir, 'test.gms')
            with open(test, 'w') as FILE:
                FILE.write(self._simple_model(n))
            result = subprocess.run(
                [solver_exec, test, "curdir=" + tmpdir, 'lo=0'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return not result.returncode
        finally:
            shutil.rmtree(tmpdir)
        return False

    def _simple_model(self, n):
        return """
            option limrow = 0;
            option limcol = 0;
            option solprint = off;
            set I / 1 * %s /;
            variables ans;
            positive variables x(I);
            equations obj;
            obj.. ans =g= sum(I, x(I));
            model test / all /;
            solve test using lp minimizing ans;
            """ % (
            n,
        )

    def version(self, config=None):
        if config is None:
            config = self.config
        pth = config.executable.path()
        if self._version_cache is None or self._version_cache[0] != pth:
            if pth is None:
                self._version_cache = (None, None)
            else:
                cmd = [pth, "audit", "lo=3"]
                subprocess_results = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                version = subprocess_results.stdout.splitlines()[0]
                version = [char for char in version.split(' ') if len(char) > 0][1]
                self._version_cache = (pth, version)
        
        return self._version_cache[1]

    def _rewrite_path_win8p3(self, path):
        """
        Return the 8.3 short path on Windows; unchanged elsewhere.

        This change is in response to Pyomo/pyomo#3579 which reported
        that GAMS (direct) fails on Windows if there is a space in
        the path. This utility converts paths to their 8.3 short-path version
        (which never have spaces).
        """
        if not sys.platform.startswith("win"):
            return str(path)

        import ctypes, ctypes.wintypes as wt

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [wt.LPCWSTR, wt.LPWSTR, wt.DWORD]

        # the file must exist, or Windows will not create a short name
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).touch(exist_ok=True)

        buf = ctypes.create_unicode_buffer(260)
        if GetShortPathNameW(str(path), buf, 260):
            return buf.value
        return str(path)


    @document_kwargs_from_configdict(CONFIG)
    def solve(self, model, **kwds):
        ####################################################################
        # Presolve
        ####################################################################
        # Begin time tracking
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)

        # Update configuration options, based on keywords passed to solve
        config: GAMSConfig = self.config(value=kwds)

        # Check if solver is available, unavailable solver error will be raised in available()
        self.available(config)
        if config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = config.timer
        StaleFlagManager.mark_all_as_stale()
        
        # Because GAMS changes the CWD when running the solver, we need
        # to convert user-provided file names to absolute paths
        # (relative to the current directory)
        if config.logfile is not None:
            config.logfile = os.path.abspath(config.logfile)

        # local variable to hold the working directory name and flags
        newdir = False
        dname = None 
        lst = "output.lst"
        output_filename = None
        with TempfileManager.new_context() as tempfile:
            # IMPORTANT - only delete the whole tmpdir if the solver was the one
            # that made the directory. Otherwise, just delete the files the solver
            # made, if not keepfiles. That way the user can select a directory
            # they already have, like the current directory, without having to
            # worry about the rest of the contents of that directory being deleted.
            if config.working_dir is None:
                dname = tempfile.mkdtemp()
                newdir = True
            else:
                dname = config.working_dir
                newdir = True
            if not os.path.exists(dname):
                os.mkdir(dname)
            basename = os.path.join(dname, model.name)
            output_filename = basename + '.gms'
            lst_filename = os.path.join(dname, lst)
            with open(
                output_filename, 'w', newline='\n', encoding='utf-8'
            ) as gms_file:
                timer.start('write_gms_file')
                self._writer.config.set_value(config.writer_config)
                gms_info = self._writer.write(
                    model,
                    gms_file,
                    symbolic_solver_labels=config.symbolic_solver_labels,
                    )
                # NOTE: omit InfeasibleConstraintException for now
                timer.stop('write_gms_file')
        if config.writer_config.put_results_format == 'gdx':
            results_filename = os.path.join(dname, f"{model.name}_p.gdx")
            statresults_filename = os.path.join(dname, "%s_s.gdx" % (config.writer_config.put_results,))
        else:
            raise NotImplementedError(
                "Only GDX format is currently supported for results."
            )
        ####################################################################
        # Apply solver
        ####################################################################
        exe_path = config.executable.path()
        command = [exe_path, output_filename, "o=" + lst, "curdir=" + dname]

        if config.tee and not config.logfile:
            # default behaviour of gams is to print to console, for
            # compatibility with windows and *nix we want to explicitly log to
            # stdout (see https://www.gams.com/latest/docs/UG_GamsCall.html)
            command.append("lo=3")
        elif not config.tee and not config.logfile:
            command.append("lo=0")
        elif not config.tee and config.logfile:
            command.append("lo=2")
        elif config.tee and config.logfile:
            command.append("lo=4")
        if config.logfile:
            command.append(f"lf={self._rewrite_path_win8p3(config.logfile)}")
        try:
            ostreams = [StringIO()]
            if config.tee:
                ostreams.append(sys.stdout)
            with TeeStream(*ostreams) as t:
                timer.start('subprocess')
                subprocess_result = subprocess.run(command, stdout=t.STDOUT, stderr=t.STDERR)
                timer.stop('subprocess')
            rc = subprocess_result.returncode
            txt = ostreams[0].getvalue()
            if config.working_dir:
                print("\nGAMS WORKING DIRECTORY: %s\n" % config.working_dir)

            if rc == 1 or rc == 127:
                raise IOError("Command 'gams' was not recognized")
            elif rc != 0:
                if rc == 3:
                    # Execution Error
                    # Run check_expr_evaluation, which errors if necessary
                    print('Error rc=3, to be determined later')
                # If nothing was raised, or for all other cases, raise this
                logger.error(
                    "GAMS encountered an error during solve. "
                    "Check listing file for details."
                )
                logger.error(txt)
                if os.path.exists(lst_filename):
                    with open(lst_filename, 'r') as FILE:
                        logger.error("GAMS Listing file:\n\n%s" % (FILE.read(),))
                raise RuntimeError(
                    "GAMS encountered an error during solve. "
                    "Check listing file for details."
                )
            if config.writer_config.put_results_format == 'gdx':
                timer.start('parse_gdx')
                model_soln, stat_vars = self._parse_gdx_results(
                    config, results_filename, statresults_filename
                )
                timer.stop('parse_gdx')

            else:
                # model_soln, stat_vars = self._parse_dat_results(
                #     results_filename, statresults_filename
                # )
                raise NotImplementedError(
                    "Only GDX format is currently supported for results."
                )
        finally:
            if not config.working_dir:
                print('Cleaning up temporary directory is handled by `release` from pyomo.common.tempfiles')

        # NOTE: solve completion time 

        ####################################################################
        # Postsolve (WIP)
        ####################################################################
        
        # Mapping between old and new contrib results
        rev_legacy_termination_condition_map = {v: k for k, v in legacy_termination_condition_map.items()}
        # rev_legacy_solver_status_map = {v: k for k, v in legacy_solver_status_map.items()}

        model_suffixes = list(
                name
                for (
                    name,
                    comp,
                ) in pyomo.core.base.suffix.active_import_suffix_generator(model)
            )
        extract_dual = 'dual' in model_suffixes
        extract_rc = 'rc' in model_suffixes
        results = GAMSResults()
        results.solver_name = "GAMS "
        results.solver_version = str(self.version())
        # results.name = output_filename
        # results.lower_bound = stat_vars["OBJEST"]
        # results.upper_bound = stat_vars["OBJEST"]
        # results.number_of_variables = stat_vars["NUMVAR"]
        # results.number_of_constraints = stat_vars["NUMEQU"]
        # results.number_of_nonzeros = stat_vars["NUMNZ"]
        # results.number_of_binary_variables = None
        # # Includes binary vars:
        # results.number_of_integer_variables = stat_vars["NUMDVAR"]
        # results.number_of_continuous_variables = (
        #     stat_vars["NUMVAR"] - stat_vars["NUMDVAR"]
        # )
        # results.number_of_objectives = 1  # required by GAMS writer
        # obj = list(model.component_data_objects(Objective, active=True))
        # assert len(obj) == 1, 'Only one objective is allowed.'

        solvestat = stat_vars["SOLVESTAT"]
        if solvestat == 1:
            results.gams_solver_status = SolverStatus.ok
        elif solvestat == 2:
            results.gams_solver_status = SolverStatus.ok
            results.gams_termination_condition = TerminationCondition.maxIterations
        elif solvestat == 3:
            results.gams_solver_status = SolverStatus.ok
            results.gams_termination_condition = TerminationCondition.maxTimeLimit
        elif solvestat == 5:
            results.gams_solver_status = SolverStatus.ok
            results.gams_termination_condition = TerminationCondition.maxEvaluations
        elif solvestat == 7:
            results.gams_solver_status = SolverStatus.aborted
            results.gams_termination_condition = (
                TerminationCondition.licensingProblems
            )
        elif solvestat == 8:
            results.gams_solver_status = SolverStatus.aborted
            results.gams_termination_condition = TerminationCondition.userInterrupt
        elif solvestat == 10:
            results.gams_solver_status = SolverStatus.error
            results.gams_termination_condition = TerminationCondition.solverFailure
        elif solvestat == 11:
            results.gams_solver_status = SolverStatus.error
            results.gams_termination_condition = (
                TerminationCondition.internalSolverError
            )
        elif solvestat == 4:
            results.gams_solver_status = SolverStatus.warning
            results.message = "Solver quit with a problem (see LST file)"
        elif solvestat in (9, 12, 13):
            results.gams_solver_status = SolverStatus.error
        elif solvestat == 6:
            results.gams_solver_status = SolverStatus.unknown
        
        modelstat = stat_vars["MODELSTAT"]
        if modelstat == 1:
            results.gams_termination_condition = TerminationCondition.optimal
            results.solution_status = SolutionStatus.optimal
        elif modelstat == 2:
            results.gams_termination_condition = TerminationCondition.locallyOptimal
            results.solution_status = SolutionStatus.feasible
        elif modelstat in [3, 18]:
            results.gams_termination_condition = TerminationCondition.unbounded
            # results.solution_status = SolutionStatus.unbounded
            results.solution_status = SolutionStatus.noSolution

        elif modelstat in [4, 5, 6, 10, 19]:
            results.gams_termination_condition = TerminationCondition.infeasibleOrUnbounded
            results.solution_status = SolutionStatus.infeasible
        elif modelstat == 7:
            results.gams_termination_condition = TerminationCondition.feasible
            results.solution_status = SolutionStatus.feasible
        elif modelstat == 8:
            # 'Integer solution model found'
            results.gams_termination_condition = TerminationCondition.optimal
            results.solution_status = SolutionStatus.optimal
        elif modelstat == 9:
            results.gams_termination_condition = (
                TerminationCondition.intermediateNonInteger
            )
            results.solution_status = SolutionStatus.noSolution
        elif modelstat == 11:
            # Should be handled above, if modelstat and solvestat both
            # indicate a licensing problem
            if results.gams_termination_condition is None:
                results.gams_termination_condition = (
                    TerminationCondition.licensingProblems
                )
            results.solution_status = SolutionStatus.noSolution
            # results.solution_status = SolutionStatus.error

        elif modelstat in [12, 13]:
            if results.gams_termination_condition is None:
                results.gams_termination_condition = TerminationCondition.error
            results.solution_status = SolutionStatus.noSolution
            # results.solution_status = SolutionStatus.error

        elif modelstat == 14:
            if results.gams_termination_condition is None:
                results.gams_termination_condition = TerminationCondition.noSolution
            results.solution_status = SolutionStatus.noSolution
            # results.solution_status = SolutionStatus.unknown

        elif modelstat in [15, 16, 17]:
            # Having to do with CNS models,
            # not sure what to make of status descriptions
            results.gams_termination_condition = TerminationCondition.optimal
            results.solution_status = SolutionStatus.noSolution
        else:
            # This is just a backup catch, all cases are handled above
            results.solution_status = SolutionStatus.noSolution

        # ensure backward compatibility before feeding to contrib.solver
        results.termination_condition = rev_legacy_termination_condition_map[results.gams_termination_condition]
        obj = list(model.component_data_objects(Objective, active=True))
        assert len(obj) == 1, 'Only one objective is allowed.'
        if (
            results.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}
        ):
            results.solution_loader = GMSSolutionLoader(gdx_data=model_soln, gms_info=gms_info)

            results.incumbent_objective = stat_vars["OBJVAL"]

            if config.load_solutions:
                results.solution_loader.load_vars()
                # if (
                #     hasattr(model, 'dual')
                #     and isinstance(model.dual, Suffix)
                #     and model.dual.import_enabled()
                # ):
                #     model.dual.update(results.solution_loader.get_duals())
                # if (
                #     hasattr(model, 'rc')
                #     and isinstance(model.rc, Suffix)
                #     and model.rc.import_enabled()
                # ):
                #     model.rc.update(results.solution_loader.get_reduced_costs())
            else:

                results.incumbent_objective = value(
                    replace_expressions(
                        obj[0].expr,
                        substitution_map={
                            id(v): val
                            for v, val in results.solution_loader.get_primals().items()
                        },
                        descend_into_named_expressions=True,
                        remove_named_expressions=True,
                    )
                )
        # results.solution_loader = SolSolutionLoader(None, None)
        # objctvval = stat_vars["OBJVAL"]

        # if (
        #     results.gams_solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}
        # ):
        #     results.incumbent_objective = stat_vars["OBJVAL"]
        
        # # NOTE: LegacySolverWrapper needs to modify - number_of_constraints/variables are not modifiable
        # soln = Solution()
        # has_rc_info = True
        # for sym, obj in gms_info.var_symbol_map.bySymbol.items():
        #     if obj.parent_component().ctype is Objective:
        #         soln.objective[sym] = {'Value': objctvval}
        #     if obj.parent_component().ctype is not Var:
        #         continue
        #     try:
        #         rec = model_soln[sym]
        #     except KeyError:
        #         # no solution returned
        #         rec = (float('nan'), float('nan'))
        #     # obj.value = float(rec[0])
        #     soln.variable[sym] = {"Value": float(rec[0])}
        #     if extract_rc and has_rc_info:
        #         try:
        #             # model.rc[obj] = float(rec[1])
        #             soln.variable[sym]['rc'] = float(rec[1])
        #         except ValueError:
        #             # Solver didn't provide marginals
        #             has_rc_info = False
        
        # results.soln = soln

        ####################################################################
        # Postsolve (OLD GAMS)
        ####################################################################
        # model_suffixes = list(
        #         name
        #         for (
        #             name,
        #             comp,
        #         ) in pyomo.core.base.suffix.active_import_suffix_generator(model)
        #     )
        # extract_dual = 'dual' in model_suffixes
        # extract_rc = 'rc' in model_suffixes
        # results = SolverResults()
        # results.problem.name = output_filename
        # results.problem.lower_bound = stat_vars["OBJEST"]
        # results.problem.upper_bound = stat_vars["OBJEST"]
        # results.problem.number_of_variables = stat_vars["NUMVAR"]
        # results.problem.number_of_constraints = stat_vars["NUMEQU"]
        # results.problem.number_of_nonzeros = stat_vars["NUMNZ"]
        # results.problem.number_of_binary_variables = None
        # # Includes binary vars:
        # results.problem.number_of_integer_variables = stat_vars["NUMDVAR"]
        # results.problem.number_of_continuous_variables = (
        #     stat_vars["NUMVAR"] - stat_vars["NUMDVAR"]
        # )
        # results.problem.number_of_objectives = 1  # required by GAMS writer
        # obj = list(model.component_data_objects(Objective, active=True))
        # assert len(obj) == 1, 'Only one objective is allowed.'
        # obj = obj[0]
        # objctvval = stat_vars["OBJVAL"]
        # results.problem.sense = obj.sense
        # if obj.is_minimizing():
        #     results.problem.upper_bound = objctvval
        # else:
        #     results.problem.lower_bound = objctvval

        # results.solver.name = "GAMS " + str(self.version())  

        # # Init termination condition to None to give preference to this first
        # # block of code, only set certain TC's below if it's still None
        # results.solver.termination_condition = None
        # results.solver.message = None
        # solvestat = stat_vars["SOLVESTAT"]
        # if solvestat == 1:
        #     results.solver.status = SolverStatus.ok
        # elif solvestat == 2:
        #     results.solver.status = SolverStatus.ok
        #     results.solver.termination_condition = TerminationCondition.maxIterations
        # elif solvestat == 3:
        #     results.solver.status = SolverStatus.ok
        #     results.solver.termination_condition = TerminationCondition.maxTimeLimit
        # elif solvestat == 5:
        #     results.solver.status = SolverStatus.ok
        #     results.solver.termination_condition = TerminationCondition.maxEvaluations
        # elif solvestat == 7:
        #     results.solver.status = SolverStatus.aborted
        #     results.solver.termination_condition = (
        #         TerminationCondition.licensingProblems
        #     )
        # elif solvestat == 8:
        #     results.solver.status = SolverStatus.aborted
        #     results.solver.termination_condition = TerminationCondition.userInterrupt
        # elif solvestat == 10:
        #     results.solver.status = SolverStatus.error
        #     results.solver.termination_condition = TerminationCondition.solverFailure
        # elif solvestat == 11:
        #     results.solver.status = SolverStatus.error
        #     results.solver.termination_condition = (
        #         TerminationCondition.internalSolverError
        #     )
        # elif solvestat == 4:
        #     results.solver.status = SolverStatus.warning
        #     results.solver.message = "Solver quit with a problem (see LST file)"
        # elif solvestat in (9, 12, 13):
        #     results.solver.status = SolverStatus.error
        # elif solvestat == 6:
        #     results.solver.status = SolverStatus.unknown

        # results.solver.return_code = rc  # 0
        # # Not sure if this value is actually user time
        # # "the elapsed time it took to execute a solve statement in total"
        # results.solver.user_time = stat_vars["ETSOLVE"]
        # results.solver.system_time = None
        # results.solver.wallclock_time = None
        # results.solver.termination_message = None

        # soln = Solution()

        # modelstat = stat_vars["MODELSTAT"]
        # if modelstat == 1:
        #     results.solver.termination_condition = TerminationCondition.optimal
        #     soln.status = SolutionStatus.optimal
        # elif modelstat == 2:
        #     results.solver.termination_condition = TerminationCondition.locallyOptimal
        #     soln.status = SolutionStatus.locallyOptimal
        # elif modelstat in [3, 18]:
        #     results.solver.termination_condition = TerminationCondition.unbounded
        #     soln.status = SolutionStatus.unbounded
        # elif modelstat in [4, 5, 6, 10, 19]:
        #     results.solver.termination_condition = TerminationCondition.infeasible
        #     soln.status = SolutionStatus.infeasible
        # elif modelstat == 7:
        #     results.solver.termination_condition = TerminationCondition.feasible
        #     soln.status = SolutionStatus.feasible
        # elif modelstat == 8:
        #     # 'Integer solution model found'
        #     results.solver.termination_condition = TerminationCondition.optimal
        #     soln.status = SolutionStatus.optimal
        # elif modelstat == 9:
        #     results.solver.termination_condition = (
        #         TerminationCondition.intermediateNonInteger
        #     )
        #     soln.status = SolutionStatus.other
        # elif modelstat == 11:
        #     # Should be handled above, if modelstat and solvestat both
        #     # indicate a licensing problem
        #     if results.solver.termination_condition is None:
        #         results.solver.termination_condition = (
        #             TerminationCondition.licensingProblems
        #         )
        #     soln.status = SolutionStatus.error
        # elif modelstat in [12, 13]:
        #     if results.solver.termination_condition is None:
        #         results.solver.termination_condition = TerminationCondition.error
        #     soln.status = SolutionStatus.error
        # elif modelstat == 14:
        #     if results.solver.termination_condition is None:
        #         results.solver.termination_condition = TerminationCondition.noSolution
        #     soln.status = SolutionStatus.unknown
        # elif modelstat in [15, 16, 17]:
        #     # Having to do with CNS models,
        #     # not sure what to make of status descriptions
        #     results.solver.termination_condition = TerminationCondition.optimal
        #     soln.status = SolutionStatus.unsure
        # else:
        #     # This is just a backup catch, all cases are handled above
        #     soln.status = SolutionStatus.error

        # soln.gap = abs(results.problem.upper_bound - results.problem.lower_bound)

        # has_rc_info = True
        # for sym, obj in gms_info.var_symbol_map.bySymbol.items():
        #     if obj.parent_component().ctype is Objective:
        #         soln.objective[sym] = {'Value': objctvval}
        #     if obj.parent_component().ctype is not Var:
        #         continue
        #     try:
        #         rec = model_soln[sym]
        #     except KeyError:
        #         # no solution returned
        #         rec = (float('nan'), float('nan'))
        #     # obj.value = float(rec[0])
        #     soln.variable[sym] = {"Value": float(rec[0])}
        #     if extract_rc and has_rc_info:
        #         try:
        #             # model.rc[obj] = float(rec[1])
        #             soln.variable[sym]['rc'] = float(rec[1])
        #         except ValueError:
        #             # Solver didn't provide marginals
        #             has_rc_info = False

        # if extract_dual:
        #     for c in model.component_data_objects(Constraint, active=True):
        #         if (c.body.is_fixed()) or (not (c.has_lb() or c.has_ub())):
        #             # the constraint was not sent to GAMS
        #             continue
        #         sym = gms_info.var_symbol_map.getSymbol(c)
        #         if c.equality:
        #             try:
        #                 rec = model_soln[sym]
        #             except KeyError:
        #                 # no solution returned
        #                 rec = (float('nan'), float('nan'))
        #             try:
        #                 # model.dual[c] = float(rec[1])
        #                 soln.constraint[sym] = {'dual': float(rec[1])}
        #             except ValueError:
        #                 # Solver didn't provide marginals
        #                 # nothing else to do here
        #                 break
        #         else:
        #             # Inequality, assume if 2-sided that only
        #             # one side's marginal is nonzero
        #             # Negate marginal for _lo equations
        #             marg = 0
        #             if c.lower is not None:
        #                 try:
        #                     rec_lo = model_soln[sym + '_lo']
        #                 except KeyError:
        #                     # no solution returned
        #                     rec_lo = (float('nan'), float('nan'))
        #                 try:
        #                     marg -= float(rec_lo[1])
        #                 except ValueError:
        #                     # Solver didn't provide marginals
        #                     marg = float('nan')
        #             if c.upper is not None:
        #                 try:
        #                     rec_hi = model_soln[sym + '_hi']
        #                 except KeyError:
        #                     # no solution returned
        #                     rec_hi = (float('nan'), float('nan'))
        #                 try:
        #                     marg += float(rec_hi[1])
        #                 except ValueError:
        #                     # Solver didn't provide marginals
        #                     marg = float('nan')
        #             if not math.isnan(marg):
        #                 # model.dual[c] = marg
        #                 soln.constraint[sym] = {'dual': marg}
        #             else:
        #                 # Solver didn't provide marginals
        #                 # nothing else to do here
        #                 break

        # results.solution.insert(soln)

        # Dummy fill-in for contrib-solver
        # results.solution_status = soln.status
        # results.termination_condition = rev_legacy_termination_condition_map[results.solver.termination_condition]
        # results.lower_bound = stat_vars["OBJEST"]
        # results.upper_bound = stat_vars["OBJEST"]
        # results.objective_bound = None
        # results.incumbent_objective = None
        # results.timing_info = None

        ####################################################################
        # Finish with results
        ####################################################################
        # smap_id = id(gms_info.con_symbol_map)
        # results._smap_id = smap_id
        # results._smap = None

        # if config.load_solutions:
        #     model.solutions.load_from(results)
        #     results._smap_id = None
        #     results.solution.clear()
        # else:
        #     results._smap = model.solutions.symbol_map[smap_id]
        #     model.solutions.delete_symbol_map(smap_id)

        # postsolve_completion_time = time.time()
        # if report_timing:
        #     print(
        #         "      %6.2f seconds required for postsolve"
        #         % (postsolve_completion_time - solve_completion_time)
        #     )
        #     print(
        #         "      %6.2f seconds required total"
        #         % (postsolve_completion_time - initial_time)
        #     )
        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = (
            end_timestamp - start_timestamp
        ).total_seconds()
        results.timing_info.timer = timer
        return results

    def _parse_gdx_results(self, config, results_filename, statresults_filename):
        model_soln = dict()
        stat_vars = dict.fromkeys(
            [
                'MODELSTAT',
                'SOLVESTAT',
                'OBJEST',
                'OBJVAL',
                'NUMVAR',
                'NUMEQU',
                'NUMDVAR',
                'NUMNZ',
                'ETSOLVE',
            ]
        )

        pgdx = gdxcc.new_gdxHandle_tp()
        ret = gdxcc.gdxCreateD(pgdx, os.path.dirname(config.executable.path()), 128)
        if not ret[0]:
            raise RuntimeError("GAMS GDX failure (gdxCreate): %s." % ret[1])
        if os.path.exists(statresults_filename):
            ret = gdxcc.gdxOpenRead(pgdx, statresults_filename)
            if not ret[0]:
                raise RuntimeError("GAMS GDX failure (gdxOpenRead): %d." % ret[1])
            
            specVals = gdxcc.doubleArray(gdxcc.GMS_SVIDX_MAX)
            rc = gdxcc.gdxGetSpecialValues(pgdx, specVals)
            
            specVals[gdxcc.GMS_SVIDX_EPS] = sys.float_info.min
            specVals[gdxcc.GMS_SVIDX_UNDEF] = float("nan")
            specVals[gdxcc.GMS_SVIDX_PINF] = float("inf")
            specVals[gdxcc.GMS_SVIDX_MINF] = float("-inf")
            specVals[gdxcc.GMS_SVIDX_NA] = struct.unpack(">d", bytes.fromhex("fffffffffffffffe"))[0]
            gdxcc.gdxSetSpecialValues(pgdx, specVals)

            i = 0
            while True:
                i += 1
                ret = gdxcc.gdxDataReadRawStart(pgdx, i)
                if not ret[0]:
                    break

                ret = gdxcc.gdxSymbolInfo(pgdx, i)
                if not ret[0]:
                    break
                if len(ret) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxSymbolInfo).")
                stat = ret[1]
                if not stat in stat_vars:
                    continue

                ret = gdxcc.gdxDataReadRaw(pgdx)
                if not ret[0] or len(ret[2]) == 0:
                    raise RuntimeError("GAMS GDX failure (gdxDataReadRaw).")

                if stat in ('OBJEST', 'OBJVAL', 'ETSOLVE'):
                    stat_vars[stat] = ret[2][0]
                else:
                    stat_vars[stat] = int(ret[2][0])

            gdxcc.gdxDataReadDone(pgdx)
            gdxcc.gdxClose(pgdx)

        if os.path.exists(results_filename):
            ret = gdxcc.gdxOpenRead(pgdx, results_filename)
            if not ret[0]:
                raise RuntimeError("GAMS GDX failure (gdxOpenRead): %d." % ret[1])

            specVals = gdxcc.doubleArray(gdxcc.GMS_SVIDX_MAX)
            rc = gdxcc.gdxGetSpecialValues(pgdx, specVals)

            specVals[gdxcc.GMS_SVIDX_EPS] = sys.float_info.min
            specVals[gdxcc.GMS_SVIDX_UNDEF] = float("nan")
            specVals[gdxcc.GMS_SVIDX_PINF] = float("inf")
            specVals[gdxcc.GMS_SVIDX_MINF] = float("-inf")
            specVals[gdxcc.GMS_SVIDX_NA] = struct.unpack(">d", bytes.fromhex("fffffffffffffffe"))[0]
            gdxcc.gdxSetSpecialValues(pgdx, specVals)
            
            i = 0
            while True:
                i += 1
                ret = gdxcc.gdxDataReadRawStart(pgdx, i)
                if not ret[0]:
                    break

                ret = gdxcc.gdxDataReadRaw(pgdx)
                if not ret[0] or len(ret[2]) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxDataReadRaw).")
                level = ret[2][0]
                dual = ret[2][1]

                ret = gdxcc.gdxSymbolInfo(pgdx, i)
                if not ret[0]:
                    break
                if len(ret) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxSymbolInfo).")
                model_soln[ret[1]] = (level, dual)

            gdxcc.gdxDataReadDone(pgdx)
            gdxcc.gdxClose(pgdx)

        gdxcc.gdxFree(pgdx)
        gdxcc.gdxLibraryUnload()
        return model_soln, stat_vars
    
    # def _parse_solution(self, instream: io.TextIOBase, gms_info: GAMSWriterInfo):
    #     results = Results()
    #     res, sol_data = parse_sol_file(
    #         sol_file=instream, nl_info=nl_info, result=results
    #     )

    #     if res.solution_status == SolutionStatus.noSolution:
    #         res.solution_loader = SolSolutionLoader(None, None)
    #     else:
    #         res.solution_loader = IpoptSolutionLoader(
    #             sol_data=sol_data, nl_info=nl_info
    #         )

    #     return res