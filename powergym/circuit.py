# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

import dss as opendss
import networkx as nx
import numpy as np
import os
import pandas as pd
from pathlib import Path

class Circuits():
    def __init__(self, dss_file, 
                 batt_file='Battery.csv', 
                 RB_act_num=(33,33), 
                 dss_act=False):
        # DSS
        self.dss = opendss.DSS # the dss simulator object
        self.dss_file = dss_file # path to the dss file for the whole circuit
        self.dss_act = dss_act # whether to use OpenDSS controllers defined in the circuit file

        self.batt_file = os.path.join( Path(self.dss_file).parent, batt_file)
        if not os.path.exists(self.batt_file): self.batt_file = ''
        
        self.topology = nx.Graph()
        self.edge_obj = dict() # map from frozenset({bus1, bus2}) to the (active) object on the edge
        self.dup_edges = dict() # map from duplicate edge (if any) to the objects on the edge
        self.edge_weight = dict() # map from edge to Ymatrix. Ymatrix is symmetric/tall if the number of phases is equal/different 
        self.bus_phase = dict() # map from bus name to the number of phases.
        self.bus_obj = dict() # map from bus name to the objects(load,capacitor,batteries) on the bus
        
        # circuit element
        self.lines = dict()
        self.transformers = dict()
        self.regulators = dict()
        self.loads = dict()
        self.capacitors = dict()
        self.batteries = dict()

        # regulator and battery action dim
        self.reg_act_num, self.bat_act_num = RB_act_num
        
        # initialization
        self.initialize()
    
    def set_regulator_parameters(self, tap=1.1, mintap=0.9, maxtap=1.1):
        '''
        Set all regulator parameters to the same predefined values

        Arguments:
           tap: tap value in between mintap and maxtap
           maxtap: max tap value
           mintap: min tap value
        
        Returns: None
        '''
        # run this after Text.Command = "compile" and before ActiveCircuit.Solution.Solve()
        
        ## numtaps: number of tap values between mintap and maxtap.
        ##          reg_act_num = numtaps + 1
        numtaps = self.reg_act_num - 1
        fea = [mintap, maxtap, numtaps]
        transet = set()
        for regname in self.regulators.keys():
            self.regulators[regname].tap_feature = fea.copy()
            self.regulators[regname].tap = tap
            transet.add(regname[10:])
        
        dssTrans = self.dss.ActiveCircuit.Transformers
        if dssTrans.First == 0: return # no such kind of object
        while True:
            if dssTrans.Name in transet:
                dssTrans.Tap = tap
                dssTrans.MinTap = mintap
                dssTrans.MaxTap = maxtap 
                dssTrans.NumTaps = numtaps
            if dssTrans.Next==0: break
    
    def compile(self, disable = False):
        '''
        Compile the main dss file

        Arguments:
            disable: disable sources and loads during solve.
                     This is used when computing the admittance matrix (Ymat)

        Returns: None
        '''
        self.dss.Text.Command = "compile " + self.dss_file
        self.dss.Text.Command = "Set Maxiterations=50"
        self.dss.Text.Command = "Set Maxcontroliter=100"
        
        if disable:
            self.dss.Text.Command = 'vsource.source.enabled=no'
            self.dss.Text.Command = 'batchedit load..* enabled=no'
        else:
            self.dss.Text.Command = 'vsource.source.enabled=yes'
            self.dss.Text.Command = 'batchedit load..* enabled=yes' 
        
        if not self.dss_act:
            self.dss.Text.Command = "Set ControlMode = off"
    
    def reset(self):
        self.compile() # this include resetting regulators in dss
                       # capacitors in dss and batteries in dss
        # reset regulator, capacitors and batteries in objects
        self.set_regulator_parameters()
        self.set_all_capacitor_statuses([1]*len(self.capacitors), change_dss=False)
        for bat in self.batteries.keys():
            self.batteries[bat].reset()
        #self.dss.ActiveCircuit.Solution.Solve()
        self.dss.ActiveCircuit.Solution.SolveNoControl()

    def initialize(self, noWei=True):
        '''
        Compile and generate all data members.

        Arguments:
            noWei: don't compute the admittance matrix (Ymat), stored as self.edge_weight

        Returns: None
        '''

        if not noWei:
            self.compile(disable = True)
            self.__cal_edgeWei_busPhase(noWei=noWei)
            self.compile(disable = False)
        else:
            self.compile(disable = False)
            self.__cal_edgeWei_busPhase(noWei=noWei)

        # edges
        regulators, valid_trans2edge, line2edge = self._get_edge_name()
        self._gen_reg_obj(regulators)
        self._gen_trans_obj(valid_trans2edge)
        self._gen_line_obj(line2edge)
        self.set_regulator_parameters()
        

        # nodes
        self._gen_load_cap_obj()
        if self.batt_file != '':
            self._gen_bat_obj()
        
        #self.dss.ActiveCircuit.Solution.Solve()
        self.dss.ActiveCircuit.Solution.SolveNoControl()
        
    def get_all_capacitor_statuses(self):
        '''
        Get taps of all regulators

        Returns: 
            capacitor statuses (dict)
        '''
        states = dict()
        dssCap = self.dss.ActiveCircuit.Capacitors
        if dssCap.First == 0: return # no such object 
        while True:
            states['Capacitor.' + dssCap.Name] = dssCap.States[0]
            if dssCap.Next==0: break
        return states

    def set_all_capacitor_statuses(self, statuses, change_dss=True):
        '''
        Set statuses of all capacitors

        Arguments:
            statuses: array of 0-1 integers
        
        Returns:
            the absolute change of statuses
        '''
        assert len(statuses)>0 and len(statuses)==len(self.capacitors), 'inconsistent statuses'
        statuses = np.array(statuses, dtype=int)

        # set capacitor objects
        diff = np.zeros(len(statuses))
        cap2st = dict()
        for i, cap in enumerate(self.capacitors.keys()):
            capa = self.capacitors[cap]
            diff[i] = abs(capa.status - statuses[i])
            capa.status = statuses[i]
            cap2st[capa.name[10:]] = statuses[i]
        
        # set dss object
        if change_dss:
            dssCap = self.dss.ActiveCircuit.Capacitors
            if dssCap.First == 0: return # no such object 
            while True:
                if dssCap.Name in cap2st:
                    dssCap.States = [cap2st[dssCap.Name]]
                if dssCap.Next==0: break
        return diff

    def get_all_regulator_tapnums(self):
        '''
        Get tapnums of all regulators

        Returns: 
            regulator tapnums (dict)
        '''
        mintap, maxtap, numtaps = self.regulators[next(iter(self.regulators))].tap_feature
        step = (maxtap - mintap) / numtaps

        # get trans name
        trans = {regname[10:] for regname in self.regulators.keys()}

        # get taps from dss
        tapnums = dict()
        dssTrans = self.dss.ActiveCircuit.Transformers
        if dssTrans.First == 0: return # no such kind of object
        while True:
            if dssTrans.Name in trans:
                tapnums['Regulator.'+dssTrans.Name] = int( (dssTrans.Tap - mintap)/step )
            if dssTrans.Next==0: break
        
        return tapnums

    def set_all_regulator_tappings(self, tapnums, change_dss=True):
        '''
        Set tap values of all regulators

        Arguments:
            tapnums: array of integers in [0, numtaps]
        
        Returns: 
            the absolute change of tapnums
        '''
        assert len(tapnums)>0 and len(tapnums) == len(self.regulators), 'inconsistent tapnums'
        mintap, maxtap, numtaps = self.regulators[next(iter(self.regulators))].tap_feature
        step = (maxtap - mintap) / numtaps
        tapnums = np.maximum(0, np.minimum(numtaps, np.array(tapnums, dtype=int) ) )
        taps = tapnums*step + mintap

        # set regulator objects
        diff = np.zeros(len(taps))
        trans2tap = dict()
        for i, reg in enumerate(self.regulators.keys()):
            regu = self.regulators[reg]
            diff[i] = abs((regu.tap - taps[i])/step)
            regu.tap = taps[i]
            # remove 'Regulator.' from the head of the name
            trans2tap[reg[10:]] = (taps[i], tapnums[i]) 

        # set dss
        if change_dss:
            dssTrans = self.dss.ActiveCircuit.Transformers
            if dssTrans.First == 0: return # no such kind of object
            while True:
                if dssTrans.Name in trans2tap:
                    tap, tapnum = trans2tap[dssTrans.Name]
                    dssTrans.NumTaps = tapnum
                    dssTrans.Tap = tap
                if dssTrans.Next==0: break 
        return diff

    def set_all_batteries_before_solve(self, nkws_or_states, change_dss=True):
        '''
        Set the states of all batteries
        ( Run this function before each solve() )

        Arguments:
            nkws_or_states: array of nkws or states
                 nkw: continuous battery's normalized discharge power in [-1, 1]
                 state: discrete battery's discharge state in [0, len(avail_kw)-1]
        
        Returns:
            the absolute change of states in integers
        '''
        
        assert len(nkws_or_states)>0 and len(nkws_or_states) == len(self.batteries), 'inconsistent states'
        if self.bat_act_num == np.inf:
            nkws_or_states = np.array(nkws_or_states, dtype=np.float32)
        else:
            nkws_or_states = np.array(nkws_or_states, dtype=int)

        # set battery object
        bat2kwkvar = dict()
        for i, bat in enumerate(self.batteries.keys()):
            batt = self.batteries[bat]
            kw = batt.state_projection(nkws_or_states[i]) # projection
            kvar = kw / batt.pf
            bat2kwkvar[batt.name[8:]] = (kw, kvar) # remove the header 'Battery.'
       
        ## change kw in dss
        if change_dss:
            dssGen = self.dss.ActiveCircuit.Generators
            if dssGen.First == 0: return # no such kind of object
            while True:
                if dssGen.Name in bat2kwkvar:
                    kw, kvar = bat2kwkvar[dssGen.Name]
                    dssGen.kW = kw
                    dssGen.kvar = kvar
                if dssGen.Next==0: break
        
        ## run solve() afterward
        
    def set_all_batteries_after_solve(self):
        '''
        Update the kwh and the soc based on the actual power shown in the dss object.
        ( Run this function after each solve() )

        Arguments: None
        Returns: soc error and discharge error
        '''
        soc_errs, discharge_errs = np.zeros(len(self.batteries)), np.zeros(len(self.batteries))
        for i, bat in enumerate(self.batteries):
            batt = self.batteries[bat]
            batt.kwh += batt.actual_power() * batt.duration
            # enforce capacity constraint and round to integer
            batt.kwh = round( max(0.0, min(batt.max_kwh, batt.kwh) ) )
            batt.soc = batt.kwh / batt.max_kwh
            soc_errs[i] = abs(batt.soc - batt.initial_soc)
           
            if self.bat_act_num == np.inf:
                discharge_errs[i] = max(0.0, batt.kw)/batt.max_kw
            else:
                discharge_errs[i] = max(0.0, batt.avail_kw[batt.state])/batt.max_kw
        return soc_errs, discharge_errs

    def _get_edge_name(self):
        '''
        Compute the object names on all edges.

        Arguments: None

        Returns: 
            regulators [dict]: map from edge to all transformers and regcontrols of a regulator.
            valid_trans2edge [dict]: map from non-regulator transformers to edge
            line2edge [dict]: map from line to edge
        '''
        def get_edge(type_name, ignore_duplicate = False):
            self.dss.ActiveCircuit.SetActiveElement(type_name)
            buses = self.dss.ActiveCircuit.ActiveElement.BusNames

            # transformer may have >2 buses
            if type_name.startswith('Transformer'):
                assert len(buses) in [2,3], type_name + ' has invalid number of terminals'
            else:
                assert len(buses) == 2, type_name + ' has more than two terminals'

            if len(buses)==2:
                bus1, bus2 = map(lambda x: x.lower().split('.'), buses)
                bus1, bus2 = bus1[0], bus2[0]
                edge = frozenset({bus1,bus2})
            else:
                bus1, bus2, bus3 = map(lambda x: x.lower().split('.'), buses)
                bus1, bus2, bus3 = bus1[0], bus2[0], bus3[0]
                edge = frozenset({bus1,bus2,bus3})
            
            if ignore_duplicate:
                return edge, False
            else:
                return edge, (edge in self.edge_obj)
            
        # deal with RegControls as an exception first        
        regulators = set() # names of transformers acting as a regulator
        dssReg = self.dss.ActiveCircuit.RegControls
        if dssReg.First == 0: return # no such kind of object
        while True:
            #reg_name = dssReg.Name
            trans_name = dssReg.Transformer
            regulators.add(trans_name)
            if dssReg.Next==0: break
        
        # run for line and transformer
        valid_trans2edge= dict()
        line2edge = dict()
        for type in ['Transformer','Line']:
            if type == 'Line':
                names = self.dss.ActiveCircuit.Lines.AllNames
            elif type == 'Transformer':
                names = self.dss.ActiveCircuit.Transformers.AllNames
          
            for name in names:
                # ignore regulators since they have been processed
                if type == 'Transformer' and name in regulators: continue
                type_name = type + '.' + name
                edge, has_dup = get_edge(type_name)
                
                # handle duplicate
                if not has_dup:
                    self.edge_obj[edge] = type_name
                else:
                    if type =='Line': # allow jump circuit
                        dup_name = self.edge_obj[edge]
                        #assert not dup_name.startswith('Line'), 'Duplicated lines: {} {}'.format(dup_name, type_name)
                        self.edge_obj[edge] = type_name
                 
                    if edge not in self.dup_edges:
                        self.dup_edges[edge] = [dup_name, type_name]
                    else:
                        self.dup_edges[edge].append(type_name)
                
                if type == 'Transformer':
                    valid_trans2edge[name] = edge
                else:
                    line2edge[name] = edge
        return regulators, valid_trans2edge, line2edge
    
    def _gen_reg_obj(self, regulators):
        '''
        Generate all regulator objects:

        Arguments: 
            regulators (set): names of regulators acting as a regulator

        Returns: None
        '''
        if len(regulators)==0: return
        dssTrans = self.dss.ActiveCircuit.Transformers
        if dssTrans.First == 0: return # no such kind of object
        while True:
            name = dssTrans.Name
            if name in regulators:
                tap = [dssTrans.Tap, dssTrans.MinTap, dssTrans.MaxTap, dssTrans.NumTaps]
                fea = [dssTrans.Xhl, dssTrans.R]
                for wdg in range(1, 1+dssTrans.NumWindings):
                    dssTrans.Wdg = wdg
                    fea = fea + [dssTrans.kV, dssTrans.kVA]
                if dssTrans.NumWindings==3:
                    fea = fea + [dssTrans.Xht, dssTrans.Xlt]
                
                # get the correct bus order
                self.dss.ActiveCircuit.SetActiveElement('Transformer.'+name)
                buses = self.dss.ActiveCircuit.ActiveElement.BusNames

                self.add_regulators('Regulator.'+name, buses, fea, tap)
            if dssTrans.Next==0: break
                
    def _gen_trans_obj(self, valid_trans2edge):
        '''
        Generate all transformer objects:

        Arguments: 
            valid_trans2edge [dict]: one of the returned objects of self._get_edge_name()

        Returns: None
        '''
        dssTrans = self.dss.ActiveCircuit.Transformers
        if dssTrans.First == 0: return # no such kind of object
        while True:
            name = dssTrans.Name
            if name in valid_trans2edge:
                #tap = [dssTrans.Tap, dssTrans.MinTap, dssTrans.MaxTap, dssTrans.NumTaps]
                fea = [dssTrans.Xhl, dssTrans.R]
                for wdg in range(1, 1+dssTrans.NumWindings):
                    dssTrans.Wdg = wdg
                    fea = fea + [dssTrans.kV, dssTrans.kVA]
                if dssTrans.NumWindings==3:
                    fea = fea + [dssTrans.Xht, dssTrans.Xlt]
                
                # get the correct bus order
                self.dss.ActiveCircuit.SetActiveElement('Transformer.'+name)
                buses = self.dss.ActiveCircuit.ActiveElement.BusNames

                self.add_transformers('Transformer.'+name, buses, fea)
            if dssTrans.Next==0: break
    
    def _gen_line_obj(self, line2edge):
        '''
        Generate all line objects:

        Arguments: 
            line2edge [dict]: one of the returned objects of self._get_edge_name()

        Returns: None
        '''
        dssLine = self.dss.ActiveCircuit.Lines
        if dssLine.First == 0: return # no such kind of object
        while True:
            name = dssLine.Name
            assert name in line2edge, 'missing line for Line.{}'.format(name)
            self.dss.ActiveCircuit.SetActiveElement('Line.'+name)
            buses = self.dss.ActiveCircuit.ActiveElement.BusNames
            mats = [ dssLine.Rmatrix, dssLine.Xmatrix, dssLine.Cmatrix ]
            self.add_lines('Line.' + name, buses, mats)
            if dssLine.Next==0: break
    
    def _gen_load_cap_obj(self):
        '''
        Generate all load and capacitor objects

        Arguments: None

        Returns: None
        '''
        for type in ['Load','Capacitor']:
            if type == 'Load':
                dssObj = self.dss.ActiveCircuit.Loads
            else:
                dssObj = self.dss.ActiveCircuit.Capacitors
            if dssObj.First == 0: break # no such kind of object
            while True:
                objname = self.dss.ActiveCircuit.CktElements.Name
                BusNames = self.dss.ActiveCircuit.CktElements.BusNames[0].split('.')
                bus = BusNames[0]
                if len(BusNames)>1:
                    phases = BusNames[1:]
                else: 
                # if not specifying the phases, use all phases at the bus
                    phases = self.bus_phase[bus]
                
                if type == 'Load':
                    fea = [dssObj.kV, dssObj.kW, dssObj.kvar]
                    self.add_loads(objname, bus, phases, fea)
                else:
                    fea = [dssObj.States[0], dssObj.kV, dssObj.kvar]
                    self.add_capacitors(objname, bus, phases, fea)
                if dssObj.Next ==0: break
    
    def _gen_bat_obj(self):
        '''
        Generate all battery objects defined in self.batt_file

        Arguments: None

        Returns: None
        '''
        batt = pd.read_csv(self.batt_file, sep=',', header = 0)
        batt = batt.set_index('name')
        batts = {name:feature for name, feature in batt.iterrows()}
        
        dssGen = self.dss.ActiveCircuit.Generators
        if dssGen.First == 0: return # no such kind of object
        while True:
            name = dssGen.Name
            if name in batts:
                feature = batts[name]
                dssGen.kW = 0.0 # initialize in disconnected mode
                dssGen.PF = feature.pf
                dssGen.kvar = feature.max_kw / feature.pf
                
                BusNames = self.dss.ActiveCircuit.CktElements.BusNames[0].split('.')
                bus = BusNames[0]
                if len(BusNames)>1:
                    phases = BusNames[1:]
                else: 
                # if not specifying the phases, use all phases at the bus
                    phases = self.bus_phase[bus]
                self.add_batteries('Battery.'+name, bus, phases, feature)
            if dssGen.Next==0: break
        
    def __cal_edgeWei_busPhase(self, noWei=True):
        '''
        Calculate edge weights (admittance matrix, Ymat) and number of phases on each bus

        Arguments:
            noWei: don't compute edge weights

        Returns: None
        '''
        # sort y nodes in alphabetical order
        YNodeOrder = np.array(self.dss.Circuits.YNodeOrder)
        order = np.argsort(YNodeOrder)
        YNodeOrder = YNodeOrder[order]
        
        # find the range and bus phase
        bus_range = dict()
        for i, node_name in enumerate(YNodeOrder):
            bus_name = node_name.split('.', 1)[0].lower()
            if bus_name not in bus_range:
                bus_range[bus_name] = [i,i+1]
            else:
                bus_range[bus_name][1] = i+1

        for bus_name in bus_range.keys():
            self.dss.Circuits.SetActiveBus(bus_name)
            self.bus_phase[bus_name] = [str(i) for i in self.dss.Circuits.Buses.Nodes]
        #self.bus_phase = {bus:r[1]-r[0] for bus, r in bus_range.items()}
        
        if noWei: return
        bus_length = len(YNodeOrder)
        Y = self.dss.Circuits.SystemY
        Y = Y.reshape((bus_length, 2*bus_length))
        Y = Y[:, ::2] + 1j*Y[:, 1::2]
        Y = Y[order,:][:,order]
            
        # extract the submatrix of edge weight from Y
        for edge in self.edge_obj.keys():
            bus1, bus2 = tuple(edge)
            range1, range2 = bus_range[bus1], bus_range[bus2]
            nphase1, nphase2 = range1[1]-range1[0], range2[1]-range2[0]
            
            if nphase1 == nphase2:
                # symmetric matrix if same number of phases
                self.edge_weight[edge] = Y[range1[0]:range1[1], range2[0]:range2[1]]
            else:
                # store as a tall matrix
                if nphase1<nphase2:
                    self.edge_weight[edge] = Y[range2[0]:range2[1], range1[0]:range1[1]]
                else:
                    self.edge_weight[edge] = Y[range1[0]:range1[1], range2[0]:range2[1]]
    
    def bus_voltage(self, bus_name):
        '''
        Get voltage of a bus

        Arguments:
            bus_name [str]: the bus name

        Returns:
            per-unit voltages and angles
        '''
        return self.dss.ActiveCircuit.Buses(bus_name).puVmagAngle

    def edge_current(self, edge_obj_name):
        '''
        Get current on an edge object

        Arguments:
            edge_obj_name [str]: the name of an edge object

        Returns:
            currents values represented in real and imaginary parts.
        '''
        return self.dss.ActiveCircuit.CktElements(edge_obj_name).Currents
    
    def total_loss(self):
        # get loss in [kw, kvar]
        return self.dss.ActiveCircuit.Losses/1000
    
    def total_power(self):
        # get power in [kw, kvar]
        return self.dss.ActiveCircuit.TotalPower
    
    ########## object addition functions called by  ############
    ###           _gen_reg_obj()
    ###           _gen_trans_obj()
    ###           _gen_line_obj()
    ###           _gen_load_cap_obj()
    ###           _gen_bat_obj()
    def add_lines(self, linename, buses, mats):
        self.lines[linename] = Line(linename, buses, mats)
    
    def add_transformers(self, transname, buses, feature):
        self.transformers[transname] = Transformer(transname, buses, feature)

    def add_regulators(self, regname, buses, feature, tap):
        self.regulators[regname] = Regulator(self.dss, regname, buses, feature, tap)

    def add_capacitors(self, capname, bus, phases, feature):
        self.capacitors[capname] = Capacitor(self.dss, capname, bus, phases, feature)
        if bus not in self.bus_obj:
            self.bus_obj[bus] = [capname]
        else:
            self.bus_obj[bus].append(capname)
            
    def add_loads(self, loadname, bus, phases, feature):
        self.loads[loadname] = Load(loadname, bus, phases, feature)
        if bus not in self.bus_obj:
            self.bus_obj[bus] = [loadname]
        else:
            self.bus_obj[bus].append(loadname)

    def add_batteries(self, batname, bus, phases, feature):
        self.batteries[batname] = Battery(self.dss, batname, bus, phases, feature, \
                                          bat_act_num = self.bat_act_num)
        if bus not in self.bus_obj:
            self.bus_obj[bus] = [batname]
        else:
            self.bus_obj[bus].append(batname)
   
    ######### object addition functions end  ############
 
    
############# Edge Objects #############
class Edge():
    def __init__(self, name, bus1, bus2):
        self.name = name
        self.bus1 = bus1
        self.bus2 = bus2
    
    def __repr__(self):
        return f"Edge {self.name} at ({self.bus1}, {self.bus2}),"

class Line(Edge):
    def __init__(self, name, buses, mats):
        bus1, bus2 = map(lambda x: x.lower().split('.'), buses)
        self.phase1,self.phase2 = map(lambda b: b[1:] if len(b)>1 else ['1','2','3'], [bus1, bus2])
        bus1, bus2 = bus1[0], bus2[0]
        super().__init__(name, bus1, bus2)
        
        # The matrices are symmetric if both buses are of the same number of phases; 
        # Otherwise, the matrices are represented as a tall matrix.
        self.rmat = mats[0] # resistance matrix
        self.xmat = mats[1] # reactance matrix
        self.cmat = mats[2] # capacitance matrix


class Transformer(Edge):
     def __init__(self, name, buses, feature):
        if len(buses)==2:
            bus1, bus2 = map(lambda x: x.lower().split('.'), buses)
            phase1, phase2 = map(lambda b: b[1:] if len(b)>1 else ['1','2','3'], [bus1, bus2])
            bus1, bus2 = bus1[0], bus2[0]
        else:
            bus1, bus2, bus3 = map(lambda x: x.lower().split('.'), buses)
            phase1, phase2, phase3 = map(lambda b: b[1:] if len(b)>1 else ['1','2','3'], \
                                         [bus1, bus2, bus3])
            bus1, bus2, bus3 = bus1[0], bus2[0], bus3[0]
            self.bus3 = bus3
            self.phase3 = phase3
        super().__init__(name, bus1, bus2)
        self.phase1 = phase1
        self.phase2 = phase2

        # 2 windings: [xhl, r, kv_wdg1, kva_wdg1, kv_wdg2, kva_wdg2]
        # 3 windings: [xhl, r, kv_wdg1, kva_wdg1, kv_wdg2, kva_wdg2, kv_wdg3, kva_wdg3, xht, xlt]
        self.trans_feature = feature 
    
class Regulator(Edge):
     def __init__(self, dss, name, buses, feature, tapfea):
        assert len(buses)==2, 'invalid number of buses for ' + name
        bus1, bus2 = map(lambda x: x.lower().split('.'), buses)
        phase1, phase2 = map(lambda b: b[1:] if len(b)>1 else ['1','2','3'], [bus1, bus2])
        bus1, bus2 = bus1[0], bus2[0]
        super().__init__(name, bus1, bus2)
        self.phase1 = phase1
        self.phase2 = phase2

        self.trans_feature = feature     # [xhl, r, kv_wdg1, kva_wdg1, kv_wdg2, kva_wdg2]
        self.dss = dss                   # the circuit's dss simulator object
        self.tap = tapfea[0]             # tap value
        self.tap_feature = tapfea[1:]    # [mintap, maxtap, numtaps]


############# Node Objects #############
class Node():
    def __init__(self, name, bus1, phases):
        self.name = name
        self.bus1 = bus1
        self.phases = phases  # names of the active phases; e.g., ['1','2','3']
    
    def __repr__(self):
        return f"Node {self.name} at {self.bus1}"

class Load(Node):
    def __init__(self, loadname, bus1, phases, feature):
        super().__init__(loadname, bus1, phases)
        self.feature = feature # [kV, kW, kvar]

class Capacitor(Node):
    def __init__(self, dss, capname, bus1, phases, feature):
        super().__init__(capname, bus1, phases)
        self.dss = dss             # the circuit's dss simulator object
        self.status = feature[0]   # close: 1,  open: 0
        self.feature = feature[1:] # [kV, kvar]

    def __repr__(self):
        return f'Capacitor status: {self.status!r},\
               Voltage at Bus: {self.bus1!r}, {self.dss.ActiveCircuit.Buses[self.bus1].puVmagAngle}'
    
    def set_status(self, status):
        '''
        Set the status of this capacitor

        Arguments:
            status: the status to be set
        
        Returns:
            the absolute change of status in integer
        '''
        diff = abs(self.status - status) # record state difference
        dssCap = self.dss.ActiveCircuit.Capacitors
        if dssCap.First == 0: return # no such object 
        while True:
            if self.name.endswith(dssCap.Name):
                self.status = status
                dssCap.States = [self.status]
                break
            if dssCap.Next==0: break
        return diff
        # should re-solve self.dss later

class Battery(Node):
    def __init__(self, dss, batname, bus1, phases, feature, bat_act_num=33):
        super().__init__(batname, bus1, phases)
        self.dss = dss                     # the circuit's dss simulator object
        self.max_kw = feature.max_kw       # maximum power magnitude
        self.pf = feature.pf               # power factor
        self.max_kwh = feature.max_kwh     # capacity
        self.kwh = feature.initial_kwh     # current charge
        self.soc = self.kwh / self.max_kwh # state of charge
        self.initial_soc = self.soc
        self.duration = self.dss.ActiveCircuit.Solution.StepSize/3600.0 # time step in hour
        if self.duration <1e-5: self.duration=1.0
        
        # battery states
        self.bat_act_num = bat_act_num
        if bat_act_num == np.inf:
            ## continuous discharge state
            self.kw = 0.0
        else:
            ## finite discharge state
            ## current kw = avail_kw[state]
            ## kw > 0 means discharging
            mode_num = bat_act_num // 2
            diff = self.max_kw / mode_num
            ## avail_kw: a discrete range from -max_kw to max_kw
            self.avail_kw = [n*diff for n in range(-mode_num, mode_num+1)]
            self.state = len(self.avail_kw)//2  # initialize as disconnected mode
         
    def __repr__(self):
        return f'Battery Available kW: {self.avail_kw!r}, \
        Status: {self.state!r}, kWh: {self.kwh!r}, SOC: {self.soc!r}, \
        Actual kW: {-self.actual_power()!r}'
   
    def state_projection(self, nkw_or_state):
        '''
        Project to the valid state

        Arguments:
            nkw_or_state: nkw: continuous battery's normalized discharge power in [-1, 1]
                          state: discrete battery's discharge state in [0, len(avail_kw)-1]
        Returns:
            the valid discharge kw
        '''
        if self.bat_act_num == np.inf:
            kw = max( -1.0, min( 1.0, nkw_or_state) )*self.max_kw
            if kw > 0:
                kw = min( self.kwh/self.duration, kw)
            else:
                kw = max( (self.kwh - self.max_kwh)/self.duration, kw)
            self.kw = kw
            return kw
        else:
            state = max( 0, min(len(self.avail_kw)-1, nkw_or_state))
            mid = len(self.avail_kw)//2
            if state > mid: # discharging
                allowed_kw = self.kwh/self.duration # max kw
                if self.avail_kw[state] > allowed_kw:
                    state = int(state - np.ceil((self.avail_kw[state]-allowed_kw)/ \
                               (self.avail_kw[1]-self.avail_kw[0])-1e-8) )
            elif state < mid: # charging
                allowed_kw = (self.kwh - self.max_kwh)/self.duration # min kw
                if self.avail_kw[state] < allowed_kw:
                    state = int(state + np.ceil((allowed_kw-self.avail_kw[state])/ \
                               (self.avail_kw[1]-self.avail_kw[0])-1e-8) )
            self.state = state
            return self.avail_kw[state]

    def step_before_solve(self, nkw_or_state):
        '''
        Set the state of this battery
        ( Run this function before each solve() )

        Arguments:
            nkw_or_state: nkw: continuous battery's normalized discharge power in [-1, 1]
                          state: discrete battery's discharge state in [0, len(avail_kw)-1]
        
        Returns: None
        '''
        
        kw = self.state_projection(nkw_or_state)

        ## change kw in dss
        name = self.name[8:] # remove the header 'Battery.'
        dssGen = self.dss.ActiveCircuit.Generators
        if dssGen.First == 0: return # no such kind of object
        while True:
            if dssGen.Name == name:          
                dssGen.kW = kw
                dssGen.kvar = kw / self.pf
                break
            if dssGen.Next==0: break
        
        ## run solve() afterward
        
    def step_after_solve(self):
        '''
        Update the kwh and the soc based on the actual power shown in the dss object.
        ( Run this function after each solve() )

        Arguments: None
        Returns: soc error and discharge error
        '''
        self.kwh += self.actual_power() * self.duration
        # enforce capacity constraint and round to integer
        self.kwh = round( max(0.0, min(self.max_kwh, self.kwh) ) )
        self.soc = self.kwh / self.max_kwh
        soc_err = abs(self.soc - self.initial_soc)
        
        if self.bat_act_num == np.inf:
            discharge_err = max(0.0, self.kw)/self.max_kw
        else:
            discharge_err = max(0.0, self.avail_kw[self.state])/self.max_kw
        return soc_err, discharge_err

    def actual_power(self):
        ## get the actual power computed by dss in [-kw, -kvar]
        ## in dss, the minus means power generation
        ## so actual_power < 0 means discharging
        name = 'Generator.' + self.name[8:]
        return self.dss.ActiveCircuit.CktElements(name).TotalPowers[0]
    
    def reset(self):
        ## reset the charge
        self.soc = self.initial_soc
        self.kwh = self.soc * self.max_kwh

        ## reset to zero discharge mode
        if self.bat_act_num == np.inf:
            self.kw = 0.0
        else:
            self.state = len(self.avail_kw)//2

############ depricated #############
class MergedRegulator(Edge):
    def __init__(self, dss, name, ori_names, edge, feature):
        bus1, bus2 = tuple(edge)
        super().__init__(name, bus1, bus2)
        self.dss = dss                   # the circuit's dss simulator object
        self.tap = feature[0][0]         # tap value
        self.tap_feature = feature[0][1:]# [mintap, maxtap, numtaps]
        self.trans_feature = feature[1]  # [xhl, r, kv_wdg1, kva_wdg1, kv_wdg2, kva_wdg2]
        self.regctr_feature = feature[2] # [ForwardR, ForwardX, ForwardBand, ForwardVreg, CTPrimary, PTratio]
        self.ori_trans = []              # the transformer names associated with this regulator
        self.ori_regctr = []             # the regulator names associated with this regulator
        for trans, regctr in ori_names:
            self.ori_trans.append(trans)
            self.ori_regctr.append(regctr)

    def __repr__(self):
        return f'Reg Current Tapping: {self.tap!r}, Reg(mintap, maxtap, numtaps): {self.tap_feature!r} \
        Voltage at Bus: {self.bus1, self.dss.ActiveCircuit.Buses[self.bus1].puVmagAngle!r}, \
        Voltage at Bus: {self.bus2, self.dss.ActiveCircuit.Buses[self.bus2].puVmagAngle!r}'
    
    def set_tapping(self, numtap):
        '''
        Set the tap value of this regulator as mintap + tapnum * (maxtap-mintap)/numtaps

        Arguments:
            numtap: integer value in [0, numtaps]
        
        Returns:
            the absolute change of numtap in integers
        '''
        numtap = min( self.tap_feature[2], max(0, numtap) )
        step = (self.tap_feature[1] - self.tap_feature[0]) / self.tap_feature[2]
        new_tap = numtap*step + self.tap_feature[0]
        diff = abs((self.tap - new_tap)/step) # record tap difference
        self.tap = new_tap
        
        dssTrans = self.dss.ActiveCircuit.Transformers
        dssTrans.First
        while True:
            if dssTrans.Name in self.ori_trans:
                dssTrans.NumTaps = numtap
                dssTrans.Tap = self.tap
            if dssTrans.Next==0: break
        
        return diff
        # should re-solve self.dss later

