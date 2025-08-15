#!/usr/bin/env python3
"""
Database Schema Analyzer for IntraoralScan Reverse Engineering
Analyzes SQLite databases to understand data models and business logic
"""

import os
import sys
import sqlite3
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import hashlib

class DatabaseSchemaAnalyzer:
    def __init__(self, base_path: str = "IntraoralScan/Bin/DB", db_path: str = "analysis_results.db"):
        self.base_path = Path(base_path)
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for storing schema analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for database schema analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS database_schemas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                db_name TEXT NOT NULL,
                db_path TEXT NOT NULL,
                file_size INTEGER,
                md5_hash TEXT,
                table_count INTEGER,
                tables_info TEXT,
                relationships TEXT,
                business_logic_hints TEXT,
                confidence_score REAL,
                analysis_method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create table for table analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS table_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                db_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                column_count INTEGER,
                row_count INTEGER,
                columns_info TEXT,
                indexes_info TEXT,
                sample_data TEXT,
                business_purpose TEXT,
                confidence_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def find_database_files(self) -> List[Path]:
        """Find all database files in the DB directory"""
        db_files = []
        
        if not self.base_path.exists():
            print(f"Database directory {self.base_path} does not exist")
            return db_files
        
        # Look for SQLite database files
        for db_file in self.base_path.rglob("*"):
            if db_file.is_file():
                # Check by extension
                if db_file.suffix.lower() in ['.db', '.sqlite', '.sqlite3', '.db3']:
                    db_files.append(db_file)
                # Check by file signature (SQLite magic number)
                elif self.is_sqlite_database(db_file):
                    db_files.append(db_file)
        
        return sorted(db_files)

    def is_sqlite_database(self, file_path: Path) -> bool:
        """Check if a file is a SQLite database by reading its header"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                # SQLite database files start with "SQLite format 3\000"
                return header.startswith(b'SQLite format 3\x00')
        except Exception:
            return False

    def analyze_database(self, db_path: Path) -> Dict:
        """Analyze a SQLite database file"""
        result = {
            "db_name": db_path.name,
            "db_path": str(db_path),
            "file_size": db_path.stat().st_size,
            "md5_hash": self.get_file_hash(db_path),
            "table_count": 0,
            "tables_info": [],
            "relationships": [],
            "business_logic_hints": [],
            "confidence_score": 0.0,
            "analysis_method": "sqlite_introspection"
        }
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            result["table_count"] = len(tables)
            
            # Analyze each table
            tables_analysis = []
            for table_name in tables:
                table_info = self.analyze_table(cursor, table_name)
                tables_analysis.append(table_info)
            
            result["tables_info"] = tables_analysis
            
            # Detect relationships
            relationships = self.detect_relationships(cursor, tables)
            result["relationships"] = relationships
            
            # Extract business logic hints
            business_hints = self.extract_business_logic_hints(tables_analysis)
            result["business_logic_hints"] = business_hints
            
            # Calculate confidence score
            result["confidence_score"] = self.calculate_confidence_score(result)
            
            conn.close()
            
        except Exception as e:
            print(f"Error analyzing database {db_path.name}: {e}")
            result["confidence_score"] = 0.1
            result["analysis_method"] = "error_fallback"
            
        return result

    def analyze_table(self, cursor: sqlite3.Cursor, table_name: str) -> Dict:
        """Analyze a single table"""
        table_info = {
            "table_name": table_name,
            "column_count": 0,
            "row_count": 0,
            "columns_info": [],
            "indexes_info": [],
            "sample_data": [],
            "business_purpose": "unknown",
            "confidence_score": 0.5
        }
        
        try:
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            table_info["column_count"] = len(columns)
            
            # Process column information
            columns_info = []
            for col in columns:
                col_info = {
                    "name": col[1],
                    "type": col[2],
                    "not_null": bool(col[3]),
                    "default_value": col[4],
                    "primary_key": bool(col[5])
                }
                columns_info.append(col_info)
            
            table_info["columns_info"] = columns_info
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            table_info["row_count"] = cursor.fetchone()[0]
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            table_info["indexes_info"] = [{"name": idx[1], "unique": bool(idx[2])} for idx in indexes]
            
            # Get sample data (first 3 rows)
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_rows = cursor.fetchall()
                table_info["sample_data"] = [list(row) for row in sample_rows]
            except Exception:
                table_info["sample_data"] = []
            
            # Infer business purpose
            table_info["business_purpose"] = self.infer_table_purpose(table_name, columns_info)
            
            # Calculate confidence
            if table_info["row_count"] > 0 and table_info["column_count"] > 0:
                table_info["confidence_score"] = 0.8
            elif table_info["column_count"] > 0:
                table_info["confidence_score"] = 0.6
                
        except Exception as e:
            print(f"Error analyzing table {table_name}: {e}")
            table_info["confidence_score"] = 0.2
            
        return table_info

    def detect_relationships(self, cursor: sqlite3.Cursor, tables: List[str]) -> List[Dict]:
        """Detect foreign key relationships between tables"""
        relationships = []
        
        for table in tables:
            try:
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                fks = cursor.fetchall()
                
                for fk in fks:
                    relationship = {
                        "from_table": table,
                        "from_column": fk[3],
                        "to_table": fk[2],
                        "to_column": fk[4],
                        "relationship_type": "foreign_key"
                    }
                    relationships.append(relationship)
                    
            except Exception as e:
                print(f"Error detecting relationships for table {table}: {e}")
        
        # Also detect implicit relationships by column naming patterns
        implicit_relationships = self.detect_implicit_relationships(cursor, tables)
        relationships.extend(implicit_relationships)
        
        return relationships

    def detect_implicit_relationships(self, cursor: sqlite3.Cursor, tables: List[str]) -> List[Dict]:
        """Detect implicit relationships based on column naming patterns"""
        relationships = []
        
        # Get all columns from all tables
        table_columns = {}
        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                table_columns[table] = [(col[1], col[2]) for col in columns]
            except Exception:
                continue
        
        # Look for ID patterns
        for table, columns in table_columns.items():
            for col_name, col_type in columns:
                # Look for foreign key patterns like "user_id", "patient_id", etc.
                if col_name.endswith('_id') and col_name != 'id':
                    potential_table = col_name[:-3]  # Remove '_id'
                    
                    # Check if there's a table with similar name
                    for other_table in tables:
                        if (potential_table.lower() in other_table.lower() or 
                            other_table.lower() in potential_table.lower()):
                            
                            relationship = {
                                "from_table": table,
                                "from_column": col_name,
                                "to_table": other_table,
                                "to_column": "id",
                                "relationship_type": "implicit_foreign_key"
                            }
                            relationships.append(relationship)
                            break
        
        return relationships

    def infer_table_purpose(self, table_name: str, columns_info: List[Dict]) -> str:
        """Infer the business purpose of a table based on name and columns"""
        table_lower = table_name.lower()
        column_names = [col["name"].lower() for col in columns_info]
        
        # Dental/medical specific patterns
        if any(keyword in table_lower for keyword in ['patient', 'user', 'customer']):
            return "patient_management"
        elif any(keyword in table_lower for keyword in ['scan', 'image', 'capture']):
            return "scan_data_management"
        elif any(keyword in table_lower for keyword in ['order', 'case', 'treatment']):
            return "order_management"
        elif any(keyword in table_lower for keyword in ['tooth', 'teeth', 'dental', 'oral']):
            return "dental_analysis"
        elif any(keyword in table_lower for keyword in ['calibrat', 'config', 'setting']):
            return "system_configuration"
        elif any(keyword in table_lower for keyword in ['log', 'audit', 'history']):
            return "audit_logging"
        elif any(keyword in table_lower for keyword in ['mesh', 'model', '3d']):
            return "3d_model_storage"
        elif any(keyword in table_lower for keyword in ['ai', 'ml', 'algorithm']):
            return "ai_analysis_results"
        
        # Check column patterns
        if any(col in column_names for col in ['x', 'y', 'z', 'coordinate']):
            return "spatial_data"
        elif any(col in column_names for col in ['timestamp', 'created_at', 'updated_at']):
            return "temporal_data"
        elif any(col in column_names for col in ['path', 'filename', 'url']):
            return "file_management"
        
        return "unknown"

    def extract_business_logic_hints(self, tables_analysis: List[Dict]) -> List[str]:
        """Extract business logic hints from table analysis"""
        hints = []
        
        # Analyze table purposes
        purposes = [table["business_purpose"] for table in tables_analysis]
        purpose_counts = {}
        for purpose in purposes:
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
        
        # Generate insights
        if purpose_counts.get("patient_management", 0) > 0:
            hints.append("System manages patient data and records")
        
        if purpose_counts.get("scan_data_management", 0) > 0:
            hints.append("System stores and manages 3D scan data")
        
        if purpose_counts.get("order_management", 0) > 0:
            hints.append("System handles dental treatment orders and cases")
        
        if purpose_counts.get("ai_analysis_results", 0) > 0:
            hints.append("System stores AI/ML analysis results")
        
        if purpose_counts.get("system_configuration", 0) > 0:
            hints.append("System has configurable parameters and settings")
        
        # Analyze data volumes
        total_rows = sum(table.get("row_count", 0) for table in tables_analysis)
        if total_rows > 10000:
            hints.append(f"Database contains substantial data ({total_rows:,} total rows)")
        
        # Analyze complexity
        total_tables = len(tables_analysis)
        if total_tables > 10:
            hints.append(f"Complex database schema with {total_tables} tables")
        
        return hints

    def calculate_confidence_score(self, result: Dict) -> float:
        """Calculate confidence score for database analysis"""
        score = 0.0
        
        # Base score for successful connection
        score += 0.3
        
        # Score for table count
        table_count = result.get("table_count", 0)
        if table_count > 0:
            score += min(0.3, table_count * 0.05)
        
        # Score for relationships
        relationships = result.get("relationships", [])
        if relationships:
            score += min(0.2, len(relationships) * 0.05)
        
        # Score for business logic hints
        hints = result.get("business_logic_hints", [])
        if hints:
            score += min(0.2, len(hints) * 0.04)
        
        return min(1.0, score)

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"

    def save_analysis_results(self, db_results: List[Dict]):
        """Save analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in db_results:
            # Insert database schema analysis
            cursor.execute('''
                INSERT OR REPLACE INTO database_schemas 
                (db_name, db_path, file_size, md5_hash, table_count, 
                 tables_info, relationships, business_logic_hints, 
                 confidence_score, analysis_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result["db_name"],
                result["db_path"],
                result["file_size"],
                result["md5_hash"],
                result["table_count"],
                json.dumps(result["tables_info"]),
                json.dumps(result["relationships"]),
                json.dumps(result["business_logic_hints"]),
                result["confidence_score"],
                result["analysis_method"]
            ))
            
            # Insert table analysis
            for table_info in result.get("tables_info", []):
                cursor.execute('''
                    INSERT OR REPLACE INTO table_analysis
                    (db_name, table_name, column_count, row_count, 
                     columns_info, indexes_info, sample_data, 
                     business_purpose, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result["db_name"],
                    table_info["table_name"],
                    table_info["column_count"],
                    table_info["row_count"],
                    json.dumps(table_info["columns_info"]),
                    json.dumps(table_info["indexes_info"]),
                    json.dumps(table_info["sample_data"]),
                    table_info["business_purpose"],
                    table_info["confidence_score"]
                ))
        
        conn.commit()
        conn.close()

    def generate_schema_report(self) -> Dict:
        """Generate comprehensive database schema report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get database summary
        cursor.execute('SELECT COUNT(*) FROM database_schemas')
        total_databases = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(table_count) FROM database_schemas')
        total_tables = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT COUNT(*) FROM table_analysis')
        analyzed_tables = cursor.fetchone()[0]
        
        # Get business purpose distribution
        cursor.execute('SELECT business_purpose, COUNT(*) FROM table_analysis GROUP BY business_purpose')
        purpose_distribution = dict(cursor.fetchall())
        
        # Get high-confidence databases
        cursor.execute('''
            SELECT db_name, table_count, business_logic_hints, confidence_score 
            FROM database_schemas 
            WHERE confidence_score > 0.5 
            ORDER BY confidence_score DESC
        ''')
        high_confidence_dbs = cursor.fetchall()
        
        # Get relationships summary
        cursor.execute('''
            SELECT db_name, relationships 
            FROM database_schemas 
            WHERE relationships != '[]'
        ''')
        relationships_data = cursor.fetchall()
        
        conn.close()
        
        # Process relationships
        all_relationships = []
        for db_name, rel_json in relationships_data:
            try:
                relationships = json.loads(rel_json)
                for rel in relationships:
                    rel["database"] = db_name
                    all_relationships.append(rel)
            except Exception:
                continue
        
        return {
            "summary": {
                "total_databases_analyzed": total_databases,
                "total_tables_found": total_tables,
                "tables_analyzed": analyzed_tables,
                "business_purpose_distribution": purpose_distribution,
                "total_relationships_found": len(all_relationships)
            },
            "high_confidence_databases": [
                {
                    "db_name": name,
                    "table_count": count,
                    "business_hints": json.loads(hints) if hints else [],
                    "confidence": conf
                }
                for name, count, hints, conf in high_confidence_dbs
            ],
            "relationships": all_relationships[:20]  # Limit to first 20
        }

    def run_analysis(self) -> Dict:
        """Run complete database schema analysis"""
        print("Starting Database Schema Analysis...")
        
        # Find database files
        db_files = self.find_database_files()
        print(f"Found {len(db_files)} database files to analyze")
        
        if not db_files:
            print("No database files found in the specified directory")
            return {"summary": {"total_databases_analyzed": 0}}
        
        # Analyze each database
        db_results = []
        for i, db_file in enumerate(db_files, 1):
            print(f"[{i}/{len(db_files)}] Analyzing {db_file.name}...")
            
            try:
                result = self.analyze_database(db_file)
                db_results.append(result)
                
                print(f"  - Tables: {result['table_count']}")
                print(f"  - Relationships: {len(result['relationships'])}")
                print(f"  - Confidence: {result['confidence_score']:.2f}")
                
            except Exception as e:
                print(f"  - Error: {e}")
                continue
        
        # Save results
        print("Saving analysis results to database...")
        self.save_analysis_results(db_results)
        
        # Generate report
        print("Generating database schema report...")
        report = self.generate_schema_report()
        
        return report

def main():
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "IntraoralScan/Bin/DB"
    
    analyzer = DatabaseSchemaAnalyzer(base_path)
    report = analyzer.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("DATABASE SCHEMA ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total databases analyzed: {report['summary']['total_databases_analyzed']}")
    print(f"Total tables found: {report['summary']['total_tables_found']}")
    print(f"Tables analyzed: {report['summary']['tables_analyzed']}")
    print(f"Business purpose distribution: {report['summary']['business_purpose_distribution']}")
    print(f"Relationships found: {report['summary']['total_relationships_found']}")
    
    print(f"\nHigh-confidence databases:")
    for db in report.get('high_confidence_databases', [])[:5]:
        print(f"  {db['db_name']}: {db['table_count']} tables (confidence: {db['confidence']:.2f})")
        for hint in db['business_hints'][:3]:
            print(f"    - {hint}")
    
    print(f"\nKey relationships found:")
    for rel in report.get('relationships', [])[:5]:
        print(f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']} ({rel['relationship_type']})")
    
    # Save report to JSON
    with open("analysis_output/database_schema_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: analysis_output/database_schema_report.json")
    print("Database updated with schema analysis results.")

if __name__ == "__main__":
    main()