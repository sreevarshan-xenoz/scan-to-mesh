#!/usr/bin/env python3
"""
Architecture Documentation Generator
Generates comprehensive architecture diagrams and documentation from the unified analysis database.
"""

import sqlite3
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
import numpy as np

class ArchitectureDocumentationGenerator:
    def __init__(self, db_path="analysis_results.db", output_dir="analysis_output/architecture_docs"):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "diagrams").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        self.conn = None
        self.data_cache = {}
        
    def connect_db(self):
        """Connect to the analysis database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            print(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def load_all_data(self):
        """Load all relevant data from database into memory for processing"""
        if not self.conn:
            return False
            
        try:
            # Load executables with classification
            cursor = self.conn.execute("""
                SELECT * FROM executables 
                ORDER BY classification, name
            """)
            self.data_cache['executables'] = [dict(row) for row in cursor.fetchall()]
            
            # Load dependencies with confidence scores
            cursor = self.conn.execute("""
                SELECT d.*, e.name as executable_name, e.classification as exe_classification
                FROM dependencies d
                JOIN executables e ON d.executable_id = e.id
                WHERE d.is_found = 1
                ORDER BY e.classification, d.dependency_name
            """)
            self.data_cache['dependencies'] = [dict(row) for row in cursor.fetchall()]
            
            # Load IPC endpoints
            cursor = self.conn.execute("""
                SELECT i.*, e.name as executable_name, e.classification as exe_classification
                FROM ipc_endpoints i
                JOIN executables e ON i.executable_id = e.id
                WHERE i.confidence_score >= 0.3
                ORDER BY i.confidence_score DESC
            """)
            self.data_cache['ipc_endpoints'] = [dict(row) for row in cursor.fetchall()]
            
            # Load network endpoints
            cursor = self.conn.execute("""
                SELECT n.*, e.name as executable_name, e.classification as exe_classification
                FROM network_endpoints n
                JOIN executables e ON n.executable_id = e.id
                WHERE n.confidence_score >= 0.3
                ORDER BY n.confidence_score DESC
            """)
            self.data_cache['network_endpoints'] = [dict(row) for row in cursor.fetchall()]
            
            # Load algorithm DLLs and functions
            cursor = self.conn.execute("""
                SELECT a.*, af.function_signature, af.confidence as func_confidence
                FROM algorithm_dlls a
                LEFT JOIN algorithm_functions af ON a.dll_name = af.dll_name
                ORDER BY a.dll_name, af.function_signature
            """)
            self.data_cache['algorithm_data'] = [dict(row) for row in cursor.fetchall()]
            
            # Load hardware interfaces
            cursor = self.conn.execute("""
                SELECT * FROM hardware_interfaces
                ORDER BY interface_type, dll_name
            """)
            self.data_cache['hardware_interfaces'] = [dict(row) for row in cursor.fetchall()]
            
            print(f"Loaded data: {len(self.data_cache['executables'])} executables, "
                  f"{len(self.data_cache['dependencies'])} dependencies, "
                  f"{len(self.data_cache['ipc_endpoints'])} IPC endpoints")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def generate_process_topology_diagram(self):
        """Generate process topology diagram with confidence-scored relationships"""
        print("Generating process topology diagram...")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Color mapping for process classifications
        color_map = {
            'UI': '#FF6B6B',           # Red
            'Logic': '#4ECDC4',        # Teal  
            'Algorithm': '#45B7D1',    # Blue
            'Network': '#96CEB4',      # Green
            'Order': '#FFEAA7',        # Yellow
            'Unknown': '#DDA0DD',      # Plum
            'Utility': '#F0F0F0'       # Light gray
        }
        
        # Add nodes (executables)
        for exe in self.data_cache['executables']:
            classification = exe.get('classification', 'Unknown')
            confidence = exe.get('confidence_score', 0.5)
            
            G.add_node(exe['name'], 
                      classification=classification,
                      confidence=confidence,
                      size_mb=exe.get('size_bytes', 0) / (1024*1024),
                      color=color_map.get(classification, color_map['Unknown']))
        
        # Add edges (dependencies) with confidence weights
        dependency_counts = defaultdict(int)
        for dep in self.data_cache['dependencies']:
            exe_name = dep['executable_name']
            dep_name = dep['dependency_name']
            
            # Only add if both nodes exist
            if exe_name in G.nodes and dep_name.replace('.dll', '.exe') in G.nodes:
                target = dep_name.replace('.dll', '.exe')
                G.add_edge(exe_name, target, weight=1.0, type='dependency')
                dependency_counts[exe_name] += 1
            elif dep_name in [node for node in G.nodes]:
                G.add_edge(exe_name, dep_name, weight=1.0, type='dependency')
                dependency_counts[exe_name] += 1
        
        # Create the plot
        plt.figure(figsize=(20, 16))
        
        # Use spring layout with custom parameters
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes with different sizes based on confidence and classification
        for classification in color_map.keys():
            nodes = [n for n, d in G.nodes(data=True) if d.get('classification') == classification]
            if nodes:
                node_sizes = [max(300, G.nodes[n].get('confidence', 0.5) * 1000) for n in nodes]
                node_colors = [G.nodes[n]['color'] for n in nodes]
                
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                     node_color=node_colors, 
                                     node_size=node_sizes,
                                     alpha=0.8, edgecolors='black', linewidths=1)
        
        # Draw edges with varying thickness based on dependency count
        edge_weights = [dependency_counts.get(u, 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [max(0.5, (w / max_weight) * 3) for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                              edge_color='gray', arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Create legend
        legend_elements = [mpatches.Patch(color=color, label=f'{classification} ({len([n for n, d in G.nodes(data=True) if d.get("classification") == classification])})')
                          for classification, color in color_map.items()
                          if len([n for n, d in G.nodes(data=True) if d.get('classification') == classification]) > 0]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.title("IntraoralScan Process Topology\n(Node size = confidence, Edge thickness = dependency count)", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save diagram
        diagram_path = self.output_dir / "diagrams" / "process_topology.png"
        plt.savefig(diagram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate topology statistics
        stats = {
            'total_processes': len(G.nodes),
            'total_dependencies': len(G.edges),
            'classification_breakdown': dict(Counter([d.get('classification', 'Unknown') 
                                                    for n, d in G.nodes(data=True)])),
            'most_connected_processes': sorted([(n, len(list(G.neighbors(n)))) 
                                              for n in G.nodes], key=lambda x: x[1], reverse=True)[:10],
            'confidence_distribution': {
                'high_confidence': len([n for n, d in G.nodes(data=True) if d.get('confidence', 0) >= 0.8]),
                'medium_confidence': len([n for n, d in G.nodes(data=True) if 0.5 <= d.get('confidence', 0) < 0.8]),
                'low_confidence': len([n for n, d in G.nodes(data=True) if d.get('confidence', 0) < 0.5])
            }
        }
        
        # Save topology data
        with open(self.output_dir / "data" / "process_topology_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Process topology diagram saved to: {diagram_path}")
        return diagram_path, stats
    
    def generate_communication_matrix(self):
        """Generate communication matrices showing IPC and network flows"""
        print("Generating communication matrix...")
        
        # Prepare data for IPC communication matrix
        ipc_data = defaultdict(lambda: defaultdict(list))
        
        for ipc in self.data_cache['ipc_endpoints']:
            exe_name = ipc['executable_name']
            endpoint_type = ipc['endpoint_type']
            endpoint_name = ipc['endpoint_name']
            direction = ipc.get('direction', 'unknown')
            confidence = ipc.get('confidence_score', 0.5)
            
            ipc_data[exe_name][endpoint_type].append({
                'name': endpoint_name,
                'direction': direction,
                'confidence': confidence
            })
        
        # Create IPC matrix visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # IPC Endpoints by Process
        processes = list(ipc_data.keys())
        endpoint_types = ['named_pipe', 'shared_memory', 'socket', 'message_queue', 'event']
        
        ipc_matrix = []
        for process in processes:
            row = []
            for endpoint_type in endpoint_types:
                count = len(ipc_data[process].get(endpoint_type, []))
                row.append(count)
            ipc_matrix.append(row)
        
        if ipc_matrix:
            im1 = ax1.imshow(ipc_matrix, cmap='YlOrRd', aspect='auto')
            ax1.set_xticks(range(len(endpoint_types)))
            ax1.set_xticklabels(endpoint_types, rotation=45)
            ax1.set_yticks(range(len(processes)))
            ax1.set_yticklabels(processes)
            ax1.set_title('IPC Endpoints by Process Type')
            
            # Add text annotations
            for i in range(len(processes)):
                for j in range(len(endpoint_types)):
                    if ipc_matrix[i][j] > 0:
                        ax1.text(j, i, str(ipc_matrix[i][j]), 
                               ha='center', va='center', fontweight='bold')
            
            plt.colorbar(im1, ax=ax1, label='Number of Endpoints')
        
        # Network Endpoints Analysis
        network_data = defaultdict(lambda: defaultdict(int))
        
        for net in self.data_cache['network_endpoints']:
            exe_name = net['executable_name']
            protocol = net.get('protocol', 'unknown')
            network_data[exe_name][protocol] += 1
        
        if network_data:
            net_processes = list(network_data.keys())
            protocols = list(set(protocol for process_data in network_data.values() 
                               for protocol in process_data.keys()))
            
            net_matrix = []
            for process in net_processes:
                row = []
                for protocol in protocols:
                    count = network_data[process].get(protocol, 0)
                    row.append(count)
                net_matrix.append(row)
            
            im2 = ax2.imshow(net_matrix, cmap='Blues', aspect='auto')
            ax2.set_xticks(range(len(protocols)))
            ax2.set_xticklabels(protocols, rotation=45)
            ax2.set_yticks(range(len(net_processes)))
            ax2.set_yticklabels(net_processes)
            ax2.set_title('Network Endpoints by Protocol')
            
            # Add text annotations
            for i in range(len(net_processes)):
                for j in range(len(protocols)):
                    if net_matrix[i][j] > 0:
                        ax2.text(j, i, str(net_matrix[i][j]), 
                               ha='center', va='center', fontweight='bold')
            
            plt.colorbar(im2, ax=ax2, label='Number of Endpoints')
        
        plt.tight_layout()
        
        # Save communication matrix
        matrix_path = self.output_dir / "diagrams" / "communication_matrix.png"
        plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate communication statistics
        comm_stats = {
            'ipc_summary': {
                'total_processes_with_ipc': len(ipc_data),
                'endpoint_type_distribution': dict(Counter([ipc['endpoint_type'] 
                                                          for ipc in self.data_cache['ipc_endpoints']])),
                'direction_distribution': dict(Counter([ipc.get('direction', 'unknown') 
                                                      for ipc in self.data_cache['ipc_endpoints']])),
                'high_confidence_endpoints': len([ipc for ipc in self.data_cache['ipc_endpoints'] 
                                                if ipc.get('confidence_score', 0) >= 0.8])
            },
            'network_summary': {
                'total_processes_with_network': len(network_data),
                'protocol_distribution': dict(Counter([net.get('protocol', 'unknown') 
                                                     for net in self.data_cache['network_endpoints']])),
                'port_ranges': self._analyze_port_ranges(),
                'high_confidence_endpoints': len([net for net in self.data_cache['network_endpoints'] 
                                                if net.get('confidence_score', 0) >= 0.8])
            }
        }
        
        # Save communication data
        with open(self.output_dir / "data" / "communication_matrix_stats.json", 'w') as f:
            json.dump(comm_stats, f, indent=2)
        
        print(f"Communication matrix saved to: {matrix_path}")
        return matrix_path, comm_stats
    
    def _analyze_port_ranges(self):
        """Analyze network port usage patterns"""
        ports = []
        for net in self.data_cache['network_endpoints']:
            port = net.get('port')
            if port and isinstance(port, int):
                ports.append(port)
        
        if not ports:
            return {}
        
        return {
            'total_unique_ports': len(set(ports)),
            'port_range': f"{min(ports)}-{max(ports)}",
            'common_ports': dict(Counter(ports).most_common(10)),
            'well_known_ports': len([p for p in ports if p < 1024]),
            'registered_ports': len([p for p in ports if 1024 <= p < 49152]),
            'dynamic_ports': len([p for p in ports if p >= 49152])
        }
    
    def generate_dependency_graph(self):
        """Build interactive dependency graphs with drill-down capability"""
        print("Generating dependency graph...")
        
        # Create dependency graph
        G = nx.DiGraph()
        
        # Add nodes for all executables and DLLs
        exe_nodes = set()
        dll_nodes = set()
        
        for exe in self.data_cache['executables']:
            exe_nodes.add(exe['name'])
            G.add_node(exe['name'], 
                      type='executable',
                      classification=exe.get('classification', 'Unknown'),
                      confidence=exe.get('confidence_score', 0.5))
        
        # Add dependency edges
        for dep in self.data_cache['dependencies']:
            exe_name = dep['executable_name']
            dep_name = dep['dependency_name']
            
            # Add DLL node if not exists
            if dep_name not in G.nodes:
                dll_nodes.add(dep_name)
                G.add_node(dep_name, 
                          type='dll',
                          is_system=dep.get('is_system_library', False))
            
            # Add edge
            G.add_edge(exe_name, dep_name, 
                      dependency_type=dep.get('dependency_type', 'dll'),
                      is_found=dep.get('is_found', True))
        
        # Create hierarchical layout
        plt.figure(figsize=(24, 18))
        
        # Separate system and non-system DLLs
        system_dlls = [n for n, d in G.nodes(data=True) 
                      if d.get('type') == 'dll' and d.get('is_system', False)]
        custom_dlls = [n for n, d in G.nodes(data=True) 
                      if d.get('type') == 'dll' and not d.get('is_system', False)]
        
        # Use hierarchical layout
        pos = {}
        
        # Position executables in the center
        exe_list = list(exe_nodes)
        for i, exe in enumerate(exe_list):
            angle = 2 * 3.14159 * i / len(exe_list)
            pos[exe] = (2 * np.cos(angle), 2 * np.sin(angle))
        
        # Position custom DLLs in inner ring
        for i, dll in enumerate(custom_dlls):
            angle = 2 * 3.14159 * i / max(len(custom_dlls), 1)
            pos[dll] = (1.2 * np.cos(angle), 1.2 * np.sin(angle))
        
        # Position system DLLs in outer ring
        for i, dll in enumerate(system_dlls):
            angle = 2 * 3.14159 * i / max(len(system_dlls), 1)
            pos[dll] = (4 * np.cos(angle), 4 * np.sin(angle))
        
        # Draw nodes with different styles
        # Executables
        nx.draw_networkx_nodes(G, pos, nodelist=list(exe_nodes),
                              node_color='lightblue', node_size=800,
                              node_shape='s', alpha=0.8, edgecolors='black')
        
        # Custom DLLs
        nx.draw_networkx_nodes(G, pos, nodelist=custom_dlls,
                              node_color='lightgreen', node_size=400,
                              node_shape='o', alpha=0.7, edgecolors='darkgreen')
        
        # System DLLs
        nx.draw_networkx_nodes(G, pos, nodelist=system_dlls,
                              node_color='lightgray', node_size=200,
                              node_shape='o', alpha=0.5, edgecolors='gray')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10)
        
        # Draw labels for executables and important DLLs only
        important_nodes = list(exe_nodes) + [dll for dll in custom_dlls if 'Sn3D' in dll or 'Dental' in dll]
        labels = {n: n for n in important_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title("IntraoralScan Dependency Graph\n(Blue squares=Executables, Green circles=Custom DLLs, Gray circles=System DLLs)", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save dependency graph
        dep_graph_path = self.output_dir / "diagrams" / "dependency_graph.png"
        plt.savefig(dep_graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate dependency statistics
        dep_stats = {
            'graph_metrics': {
                'total_nodes': G.number_of_nodes(),
                'total_edges': G.number_of_edges(),
                'executables': len(exe_nodes),
                'custom_dlls': len(custom_dlls),
                'system_dlls': len(system_dlls)
            },
            'dependency_analysis': {
                'most_dependent_executables': sorted([(n, G.out_degree(n)) for n in exe_nodes], 
                                                   key=lambda x: x[1], reverse=True)[:10],
                'most_used_dlls': sorted([(n, G.in_degree(n)) for n in dll_nodes], 
                                       key=lambda x: x[1], reverse=True)[:10],
                'isolated_components': list(nx.isolates(G))
            }
        }
        
        # Save dependency data
        with open(self.output_dir / "data" / "dependency_graph_stats.json", 'w') as f:
            json.dump(dep_stats, f, indent=2)
        
        print(f"Dependency graph saved to: {dep_graph_path}")
        return dep_graph_path, dep_stats
    
    def export_structured_data(self):
        """Export all findings from unified SQLite database to structured formats"""
        print("Exporting structured data...")
        
        # Export to JSON format
        json_export = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'database_path': self.db_path,
                'total_records': sum(len(data) for data in self.data_cache.values())
            },
            'executables': self.data_cache['executables'],
            'dependencies': self.data_cache['dependencies'],
            'ipc_endpoints': self.data_cache['ipc_endpoints'],
            'network_endpoints': self.data_cache['network_endpoints'],
            'algorithm_data': self.data_cache['algorithm_data'],
            'hardware_interfaces': self.data_cache['hardware_interfaces']
        }
        
        json_path = self.output_dir / "data" / "complete_analysis_export.json"
        with open(json_path, 'w') as f:
            json.dump(json_export, f, indent=2)
        
        # Export to CSV format for spreadsheet analysis
        csv_dir = self.output_dir / "data" / "csv_exports"
        csv_dir.mkdir(exist_ok=True)
        
        for table_name, data in self.data_cache.items():
            if data:  # Only export non-empty datasets
                df = pd.DataFrame(data)
                csv_path = csv_dir / f"{table_name}.csv"
                df.to_csv(csv_path, index=False)
        
        # Create summary report
        summary = {
            'analysis_overview': {
                'total_executables': len(self.data_cache['executables']),
                'total_dependencies': len(self.data_cache['dependencies']),
                'total_ipc_endpoints': len(self.data_cache['ipc_endpoints']),
                'total_network_endpoints': len(self.data_cache['network_endpoints']),
                'algorithm_dlls': len(set(item['dll_name'] for item in self.data_cache['algorithm_data'] if item.get('dll_name'))),
                'hardware_interfaces': len(self.data_cache['hardware_interfaces'])
            },
            'confidence_metrics': {
                'high_confidence_processes': len([exe for exe in self.data_cache['executables'] 
                                                if exe.get('confidence_score', 0) >= 0.8]),
                'high_confidence_ipc': len([ipc for ipc in self.data_cache['ipc_endpoints'] 
                                          if ipc.get('confidence_score', 0) >= 0.8]),
                'high_confidence_network': len([net for net in self.data_cache['network_endpoints'] 
                                              if net.get('confidence_score', 0) >= 0.8])
            },
            'export_paths': {
                'json_export': str(json_path),
                'csv_directory': str(csv_dir),
                'diagrams_directory': str(self.output_dir / "diagrams"),
                'reports_directory': str(self.output_dir / "reports")
            }
        }
        
        summary_path = self.output_dir / "data" / "export_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Structured data exported to: {self.output_dir / 'data'}")
        return summary
    
    def run_task_7_1(self):
        """Execute task 7.1: Create comprehensive architecture diagrams"""
        print("=== Task 7.1: Creating comprehensive architecture diagrams ===")
        
        if not self.connect_db():
            return False
        
        if not self.load_all_data():
            return False
        
        try:
            # Generate all diagrams and export data
            topology_path, topology_stats = self.generate_process_topology_diagram()
            matrix_path, comm_stats = self.generate_communication_matrix()
            dep_graph_path, dep_stats = self.generate_dependency_graph()
            export_summary = self.export_structured_data()
            
            # Create comprehensive summary
            task_7_1_summary = {
                'task': '7.1 Create comprehensive architecture diagrams',
                'completion_timestamp': datetime.now().isoformat(),
                'deliverables': {
                    'process_topology_diagram': str(topology_path),
                    'communication_matrix': str(matrix_path),
                    'dependency_graph': str(dep_graph_path),
                    'structured_data_export': export_summary['export_paths']
                },
                'statistics': {
                    'topology': topology_stats,
                    'communication': comm_stats,
                    'dependencies': dep_stats
                },
                'requirements_addressed': ['6.1', '6.2']
            }
            
            # Save task summary
            summary_path = self.output_dir / "reports" / "task_7_1_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(task_7_1_summary, f, indent=2)
            
            print(f"\n=== Task 7.1 Completed Successfully ===")
            print(f"Summary saved to: {summary_path}")
            print(f"All diagrams and data exported to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error in task 7.1: {e}")
            return False
        finally:
            self.close_db()

def main():
    """Main execution function"""
    generator = ArchitectureDocumentationGenerator()
    success = generator.run_task_7_1()
    
    if success:
        print("\nTask 7.1 completed successfully!")
        return 0
    else:
        print("\nTask 7.1 failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())