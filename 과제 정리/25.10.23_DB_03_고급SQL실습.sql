USE hospital_lab; 
 
-- [1단계] JOIN 실습
-- 문제 1. INNER JOIN
-- 진료 기록과 의사 정보를 결합하여, 각 예약의 의사 이름과 전문과목을 함께 출력하세요.
SELECT a.appointment_id, d.name AS doctor_name, d.specialty, a.visit_date
FROM appointments a
INNER JOIN doctors d ON a.doctor_id = d.doctor_id;

-- 문제 2. LEFT JOIN
-- 모든 예약을 기준으로, 의사 정보가 없더라도 결과를 출력하세요.
SELECT a.appointment_id, d.name AS doctor_name, d.specialty, a.visit_date
FROM appointments a
LEFT JOIN doctors d ON a.doctor_id = d.doctor_id;

-- 문제 3. RIGHT JOIN
-- 모든 의사를 기준으로, 예약이 없는 의사도 표시되게 하세요.
SELECT d.doctor_id , d.name AS doctor_name, d.specialty, a.appointment_id, a.visit_date, a.note
FROM appointments a
RIGHT JOIN doctors d ON a.doctor_id = d.doctor_id;

-- [2단계] 집계 함수 실습
-- 문제 4. 전체 예약 건수
-- 예약 테이블에서 전체 예약 건수를 구하세요.
SELECT COUNT(*) AS total_appointments
FROM appointments

-- 문제 5. 진료한 의사 수
-- 중복 없이 진료에 참여한 의사 수를 구하세요.
SELECT COUNT(DISTINCT doctor_id) AS unique_doctors
FROM appointments

-- 문제 6. 천 번째, 마지막 진료일
-- 진료 날짜 중 가장 빠른 날과 가장 늦은 날을 출력하세요.
SELECT MIN(visit_date) AS first_visit, MAX(visit_date) AS last_visit
FROM appointments

-- [3단계] GROUP BY 실습
-- 문제 7. 의사별 진료 횟수
-- 각 doctor_id 별로 총 진료 횟수를 출력하세요.
SELECT doctor_id, COUNT(*) AS total_visits
FROM appointments
GROUP BY doctor_id;

-- 문제 8. 환자별 진료 횟수
-- 각 patient_id 별로 몇 번 예약했는지 출력하세요.
SELECT patient_id, COUNT(*) AS appointment_count
FROM appointments
GROUP BY patient_id;

-- 문제 9. 의사별 + 환자별 통계
-- 같은 의사와 같은 환자가 몇 번 만났는지 계산하세요.
SELECT a.doctor_id, a.patient_id, COUNT(*) AS visit_count
FROM appointments a
GROUP BY a.doctor_id, a.patient_id;

-- [4단계] HAVING 실습
-- 문제 10. 진료가 2회 이상인 의사
-- 2회 이상 예약을 가진 의사만 출력하세요.
SELECT doctor_id, 
	COUNT(*) AS total_visits
FROM appointments a
GROUP BY a.doctor_id
having total_visits >=2;

-- 문제 11. 진료가 존재하는 의사만 보기
SELECT doctor_id, 
	COUNT(*) AS total_visits
FROM appointments a
GROUP BY a.doctor_id
having COUNT(*) >0;

-- [5단계] 서브쿼리 실습
-- 문제 12. 가장 최근 예약 보기
-- 전체 중 가장 늦은 visit_date를 가진 예약을 출력하세요.
SELECT *
FROM appointments
WHERE visit_date = (
	SELECT MAX(visit_date)
    FROM appointments
);

-- 문제 13. 예약이 있는 환자만 보기
-- appointments에 존재하는 patient_id만 골라 patients 테이블에서 이름을 출력하세요.
SELECT name
FROM patients
WHERE patient_id IN (
	SELECT patient_id
    FROM appointments
);

-- 문제 14. 예약이 없는 환자 찾기
-- appointments에 없는 환자를 출력하세요.
SELECT name
FROM patients
WHERE patient_id NOT IN (
	SELECT patient_id
    FROM appointments
);

-- 문제 15. 임시 테이블처럼 쓰는 서브쿼리
-- 의사별 진료 횟수를 구한 뒤, 진료가 2회 이상인 의사만 출력하세요. 


-- 문제 16. SELECT 절에 서브쿼리 사용
-- 각 의사 이름 옆에 총 진료 검수를 함께 표시하세요. 