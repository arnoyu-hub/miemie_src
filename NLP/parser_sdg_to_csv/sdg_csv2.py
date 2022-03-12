import pymongo
import pymysql
import csv
# from main.CONFIG_READER.read import get_details

class SDG_CSV_RESULTS2():
    
    def __init__(self):
            self.client = '127.0.0.1'
            self.database = '123'
            self.port = 3306
            self.username = 'yzyucl'
            self.password = 'Yzy8588903'
            self.driver = '{MySQL ODBC 8.0 Unicode Driver}'
            # self.database = get_details("SQL_SERVER", "database")
            # self.username = get_details("SQL_SERVER", "username")
            # self.client = get_details("SQL_SERVER", "client")
            # self.password = get_details("SQL_SERVER", "password")
            # self.port = 3306
            self.Faculty = ['Faculty of Arts and Humanities','Faculty of Social and Historical Sciences','Faculty of Brain Sciences','Faculty of Life Sciences','Faculty of the Built Environment', 'School of Slavonic and Eastern European Studies'
                   ,'Institute of Education', 'Faculty of Engineering Science',' Faculty of Maths & Physical Sciences', 'Faculty of Medical Sciences','Faculty of Pop Health Sciences',' Faculty of Laws']
            self.sdg_goals_no_regex = ['SDG 1','SDG 2','SDG 3','SDG 4','SDG 5','SDG 6','SDG 7','SDG 8','SDG 9','SDG 10','SDG 11','SDG 12','SDG 13','SDG 14','SDG 15','SDG 16','SDG 17']  
            self.sdg_goals = ['.*SDG 1".*','.*SDG 2.*','.*SDG 3.*','.*SDG 4.*','.*SDG 5.*','.*SDG 6.*','.*SDG 7.*','.*SDG 8.*','.*SDG 9.*','.*SDG 10.*','.*SDG 11.*','.*SDG 12.*','.*SDG 13.*','.*SDG 14.*','.*SDG 15.*','.*SDG 16.*','.*SDG 17.*']
    def generate_csv_file(self):
        con_mongo = pymongo.MongoClient('localhost', 27017)
        con_sql = pymysql.connect(host=self.client, port=self.port, db=self.database, user=self.username, password=self.password)
        cursor = con_sql.cursor(pymysql.cursors.DictCursor)
        db = con_mongo.miemie
        collection = db.MatchedModules
        with open("main/NLP/parser_sdg_to_csv/sdg_csv.csv","w+",encoding='utf-8') as file:
             csv_writer = csv.writer(file)
             for a in self.Faculty:
                csv_writer.writerow([a])
                for b in range(0,len(self.sdg_goals)):
                #--------------------------------------------------------------
                    sdg_file1 = self.sdg_goals[b]
                    sdg_file = sdg_file1.replace(" ",'')
                    if len(sdg_file) == 8:
                        sdg_file = sdg_file[2:6]
                    elif len(sdg_file) == 9 and sdg_file[6] != "\"":
                        sdg_file = sdg_file[2:7]
                    else:
                        sdg_file = sdg_file[2:6]
                    sdg_list_id = []
                    result = collection.find({"Related_SDG": {'$regex': self.sdg_goals[b]}})
                    for i in result:
                        sdg_list_id.append(i["Module_ID"])
                    sdg_list_faculty = []
                    sql = "SELECT * FROM moduledata"
                    # execute SQL
                    cursor.execute(sql)
                    # get SQL data
                    results = cursor.fetchall()
                    for row in results:
                        id = row['Module_ID']
                        faculty = row['Faculty']
                        for i in sdg_list_id:
                            if i == row['Module_ID']:
                                sdg_list_faculty.append(faculty)   
                    #-----------------------------------------------------------------------------
                    csv_writer.writerow([self.sdg_goals_no_regex[b],sdg_list_faculty.count(a)])
        # close SQL
        con_sql.close()
    def run(self):
        self.generate_csv_file()
        
                