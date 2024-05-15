from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

@app.route('/delete', methods=['POST'])
def delete_user():
    user_id = request.form.get('user_id')
    try:
        conn = sqlite3.connect('user_database.db')
        c = conn.cursor()
        c.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        return 'User deleted successfully', 200
    except Exception as e:
        return str(e), 500

@app.route('/')
def index():
    conn = sqlite3.connect('user_database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    users = c.fetchall()
    conn.close()
    return render_template('admin.html', users=users)

if __name__ == '__main__':
    app.run(debug=True)

