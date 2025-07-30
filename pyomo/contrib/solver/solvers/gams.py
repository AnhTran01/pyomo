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
from pyomo.common.config import ConfigValue, document_kwargs_from_configdict, ConfigDict
from pyomo.common.errors import (
    ApplicationError,
    DeveloperError,
    InfeasibleConstraintException,
)
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.gams_writer_v2 import GAMSWriterInfo, GAMSWriter
from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.contrib.solver.common.results import (
    Results,
    TerminationCondition,
    SolutionStatus,
)
from pyomo.contrib.solver.solvers.sol_reader import parse_sol_file, SolSolutionLoader
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoSolutionError,
)
from pyomo.opt.base.solvers import _extract_version
from pyomo.common.tee import TeeStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix
from pyomo.common.collections import ComponentMap
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
        self.solnfile: ConfigDict = self.declare(
            'solnfile', ConfigValue(
                default='GAMS_MODEL_p',
                description="Use to write out the result",
            ),
        )
        self.writer_config: ConfigDict = self.declare(
            'writer_config', GAMSWriter.CONFIG()
        )
class GAMS(SolverBase):
    # REQUIRED METHODS BY SolverBase
    # 1) available
    # 2) solve
    # 3) version
    # 4) is_persistent

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
                results = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                version = results.stdout.splitlines()[0]
                version = [char for char in version.split(' ') if len(char) > 0][1]
                self._version_cache = (pth, version)
        
        return self._version_cache[1]

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
        
        newdir = False
        dname = None # local variable to hold the working directory name
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
            # NOTE: Do we allow overwrite of existing .gms file?
            # if os.path.exists(basename + '.gms'):
            #     raise RuntimeError(
            #         f"gms file with the same name {basename + '.gms'} already exists!"
            #     )
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
            results_filename = os.path.join(dname, "GAMS_MODEL_p.gdx")
            statresults_filename = os.path.join(dname, "%s_s.gdx" % (config.writer_config.put_results,))
        else:
            raise NotImplementedError(
                "Only GDX format is currently supported for results."
            )
        ####################################################################
        # Apply solver
        ####################################################################
        exe_path = config.executable.path()
        # NOTE: The second field must be specifically `model.gms` to generate the gdx files, why?
        command = [exe_path, output_filename, "o=" + lst, "curdir=" + dname]

        # NOTE: logfile is currently unimplemented in
        # pyomo/pyomo/contrib/solver/common/base.py
        if config.tee:
            # default behaviour of gams is to print to console, for
            # compatibility with windows and *nix we want to explicitly log to
            # stdout (see https://www.gams.com/latest/docs/UG_GamsCall.html)
            command.append("lo=3")
        # elif not config.tee and not config.logfile:
        #     command.append("lo=0")
        # elif not config.tee and config.logfile:
        #     command.append("lo=2")
        # elif config.tee and config.logfile:
        #     command.append("lo=4")
        # if config.logfile:
        #     command.append(f"lf={self._rewrite_path_win8p3(config.logfile)}")
        # NOTE: This will need to be redesign - tmpfile/tmpdir is removed after write() by __exit__ in pyomo.common.tempfiles
        if results_filename:
            pass
        try:
            ostreams = [StringIO()]
            if config.tee:
                ostreams.append(sys.stdout)
            with TeeStream(*ostreams) as t:
                # BUG: execute subprocess.run should generate a results/GAMS_MODEL_p.gdx file
                result = subprocess.run(command, stdout=t.STDOUT, stderr=t.STDERR)
            rc = result.returncode
            txt = ostreams[0].getvalue()
            if config.working_dir:
                print("\nGAMS WORKING DIRECTORY: %s\n" % config.working_dir)

            if rc == 1 or rc == 127:
                raise IOError("Command 'gams' was not recognized")
            elif rc != 0:
                if rc == 3:
                    # Execution Error
                    # Run check_expr_evaluation, which errors if necessary
                    # check_expr_evaluation(model, symbolMap, 'shell') # old GAMS
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
                model_soln, stat_vars = self._parse_gdx_results(
                    config, results_filename, statresults_filename
                )
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

        ####################################################################
        # Postsolve
        ####################################################################
        # NOTE: Need to map SolutionStatus to specified pyomo.contrib.solver.common.results
        results = Results()
        results.solver_name = "GAMS "
        results.solver_version = str(self.version())
        solvestat = stat_vars["SOLVESTAT"]
        
        # mapping of solution/solver status:
        # ok = Normal termination = feasible
        # warning = Termination with unusual condition = infeasible
        # error = Terminated internally with error = noSolution
        # unknown = An uninitialized value = noSolution
        if solvestat == 1:
            results.solution_status = SolutionStatus.feasible
        # elif solvestat == 2:
        #     results.solution_status = SolverStatus.ok
        #     results.termination_condition = TerminationCondition.maxIterations
        # elif solvestat == 3:
        #     results.solution_status = SolverStatus.ok
        #     results.termination_condition = TerminationCondition.maxTimeLimit
        # elif solvestat == 5:
        #     results.solution_status = SolverStatus.ok
        #     results.termination_condition = TerminationCondition.maxEvaluations
        # elif solvestat == 7:
        #     results.solution_status = SolverStatus.aborted
        #     results.termination_condition = (
        #         TerminationCondition.licensingProblems
        #     )
        # elif solvestat == 8:
        #     results.solution_status = SolverStatus.aborted
        #     results.termination_condition = TerminationCondition.userInterrupt
        # elif solvestat == 10:
        #     results.solution_status = SolverStatus.error
        #     results.termination_condition = TerminationCondition.solverFailure
        # elif solvestat == 11:
        #     results.solution_status = SolverStatus.error
        #     results.termination_condition = (
        #         TerminationCondition.internalSolverError
        #     )
        # elif solvestat == 4:
        #     results.solution_status = SolverStatus.warning
        #     results.solver.message = "Solver quit with a problem (see LST file)"
        # elif solvestat in (9, 12, 13):
        #     results.solution_status = SolverStatus.error
        # elif solvestat == 6:
        #     results.solution_status = SolverStatus.unknown
        
        # NOTE: IGNORE SOLVER FOR NOW
        results.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
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
